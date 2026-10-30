"""
decision_engine.py — Scene-level intelligence for Ocean Guardian.

Goes beyond per-object classification to reason about the entire scene:
  - Are marine life and trash dangerously close? → safety hold
  - Are there multiple trash items? → priority cleanup zone
  - Is the scene clear of life? → safe to proceed
  - Is everything uncertain? → scan and wait

This is the "brain" layer between vision and robot actuation.
In a real deployment, scene_report.robot_command would be sent over
ROS, MQTT, or serial to the physical robot controller.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import math
import time

from detector import Detection
from config import Config, COLORS


# ── Scene report data model ───────────────────────────────────────────────────

@dataclass
class SceneReport:
    """Full situational assessment for one frame."""
    timestamp: float = field(default_factory=time.time)

    # Counts
    marine_count: int = 0
    trash_count: int = 0
    uncertain_count: int = 0

    # Flags
    proximity_conflict: bool = False   # trash too close to marine life
    priority_zone: bool = False        # many trash items present

    # Top-level decision
    robot_command: str = "SCANNING…"
    command_color: Tuple[int, int, int] = (220, 220, 220)  # BGR

    # Risk level for the whole scene: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    scene_risk: str = "LOW"

    # Which trash detections are safe to collect this frame
    collectible: List[Detection] = field(default_factory=list)
    blocked: List[Detection] = field(default_factory=list)   # too close to life

    # Command keywords in priority order (see DecisionEngine._decide_command).
    COMMAND_KEYS = ("HOLD", "OBSERVE", "PRIORITY CLEANUP", "COLLECT",
                    "PROCEED WITH CARE", "SCANNING", "CLEAR")

    @property
    def command_key(self) -> str:
        """Plain keyword of robot_command (no emoji), e.g. "HOLD" — read-only
        view used by simulation.py to drive the simulated robot."""
        for key in self.COMMAND_KEYS:
            if key in self.robot_command:
                return key
        return "SCANNING"


# ── Decision engine ───────────────────────────────────────────────────────────

class DecisionEngine:
    """
    Evaluates the full list of detections from a single frame and returns
    a SceneReport that drives both the visual overlay and (future) robot commands.

    Robot integration note:
        scene_report.robot_command is the string you'd serialize to JSON
        and publish to a ROS topic, MQTT broker, or serial port.
        e.g.:  {"command": "COLLECT", "targets": [[x1,y1,x2,y2], ...]}
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Session-level accumulators (for end-of-session summary)
        self._session_trash_seen: int = 0
        self._session_marine_seen: int = 0
        self._session_frames: int = 0
        self._session_conflicts: int = 0
        self._start_time: float = time.time()

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(self, detections: List[Detection]) -> SceneReport:
        """Main entry point. Analyse detections and return a SceneReport."""
        report = SceneReport()
        self._session_frames += 1

        # ── 1. Count categories ───────────────────────────────────────────────
        marine   = [d for d in detections if d.category == "marine"]
        trash    = [d for d in detections if d.category == "trash"]
        uncertain = [d for d in detections if d.category == "uncertain"]

        report.marine_count   = len(marine)
        report.trash_count    = len(trash)
        report.uncertain_count = len(uncertain)

        self._session_trash_seen  += len(trash)
        self._session_marine_seen += len(marine)

        # ── 2. Proximity conflict check ───────────────────────────────────────
        # If any trash bbox is within PROXIMITY_SAFETY_MARGIN_PX pixels of a
        # marine life bbox center, the trash is blocked from collection.
        blocked_trash: List[Detection] = []
        safe_trash: List[Detection] = []

        for t in trash:
            if self._is_near_marine(t, marine):
                blocked_trash.append(t)
            else:
                safe_trash.append(t)

        if blocked_trash:
            report.proximity_conflict = True
            self._session_conflicts += 1

        report.collectible = safe_trash
        report.blocked     = blocked_trash

        # ── 3. Priority zone flag ─────────────────────────────────────────────
        report.priority_zone = (
            len(safe_trash) >= self.cfg.PRIORITY_ZONE_THRESHOLD
        )

        # ── 4. Scene risk assessment ──────────────────────────────────────────
        report.scene_risk = self._assess_risk(report)

        # ── 5. Robot command ──────────────────────────────────────────────────
        report.robot_command, report.command_color = self._decide_command(report)

        return report

    def print_status(self, report: SceneReport, frame: int):
        """Print a concise one-line status to the terminal."""
        elapsed = time.time() - self._start_time
        print(
            f"  Frame {frame:>5} | "
            f"🐠 Marine: {report.marine_count}  "
            f"♻️  Trash: {report.trash_count}  "
            f"⚠️  Uncertain: {report.uncertain_count}  "
            f"| Risk: {report.scene_risk:<8} "
            f"| CMD: {report.robot_command}"
        )

    def print_summary(self):
        """Print end-of-session statistics."""
        elapsed = time.time() - self._start_time
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                   SESSION SUMMARY                            ║
╠══════════════════════════════════════════════════════════════╣
║  Duration       : {elapsed:>8.1f}s                              ║
║  Frames         : {self._session_frames:>8}                                 ║
║  Total trash    : {self._session_trash_seen:>8} detections                  ║
║  Marine sightings: {self._session_marine_seen:>7} detections                  ║
║  Safety conflicts: {self._session_conflicts:>7} events (trash near life)     ║
╚══════════════════════════════════════════════════════════════╝""")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _is_near_marine(
        self,
        trash_det: Detection,
        marine_dets: List[Detection]
    ) -> bool:
        """
        Returns True if the trash detection's bounding box is within
        PROXIMITY_SAFETY_MARGIN_PX pixels (Euclidean, center-to-center)
        of any marine life detection.

        Also returns True if the bounding boxes overlap at all (IoU > 0),
        since that would make collection 100% unsafe.
        """
        margin = self.cfg.PROXIMITY_SAFETY_MARGIN_PX
        tx1, ty1, tx2, ty2 = trash_det.bbox
        tcx, tcy = trash_det.center

        for m in marine_dets:
            # Check 1: Euclidean center distance
            mcx, mcy = m.center
            dist = math.sqrt((tcx - mcx) ** 2 + (tcy - mcy) ** 2)
            if dist < margin:
                return True

            # Check 2: Bounding box overlap (IoU > 0)
            mx1, my1, mx2, my2 = m.bbox
            inter_x = max(0, min(tx2, mx2) - max(tx1, mx1))
            inter_y = max(0, min(ty2, my2) - max(ty1, my1))
            if inter_x > 0 and inter_y > 0:
                return True

        return False

    def _assess_risk(self, r: SceneReport) -> str:
        if r.proximity_conflict:
            return "CRITICAL"
        if r.marine_count > 0 and r.trash_count > 0:
            return "HIGH"
        if r.trash_count >= self.cfg.PRIORITY_ZONE_THRESHOLD:
            return "HIGH"
        if r.trash_count > 0:
            return "MEDIUM"
        if r.uncertain_count > 0:
            return "LOW"
        return "CLEAR"

    def _decide_command(
        self,
        r: SceneReport
    ) -> Tuple[str, Tuple[int, int, int]]:
        """
        Returns (command_string, BGR_color) for the HUD.

        ┌─────────────────────────────────────────────────────┐
        │  Priority order (highest → lowest)                  │
        │  1. Proximity conflict  → HOLD (safety first)       │
        │  2. Marine only         → IGNORE / OBSERVE          │
        │  3. Priority zone       → PRIORITY CLEANUP          │
        │  4. Safe trash exists   → COLLECT                   │
        │  5. Uncertain only      → AWAIT CONFIRMATION        │
        │  6. Nothing detected    → SCANNING                  │
        └─────────────────────────────────────────────────────┘
        """
        c = COLORS

        if r.proximity_conflict:
            return ("⛔  HOLD — Marine Life Nearby", c["critical"])

        if r.marine_count > 0 and r.trash_count == 0:
            return ("🐠  OBSERVE — Marine Life Detected", c["marine"])

        if r.priority_zone:
            return ("🚨  PRIORITY CLEANUP ZONE", c["trash"])

        if len(r.collectible) > 0 and r.marine_count == 0:
            n = len(r.collectible)
            return (f"♻️   COLLECT — {n} Trash Item{'s' if n > 1 else ''} Targeted", c["trash"])

        if len(r.collectible) > 0 and r.marine_count > 0:
            # Trash present but not proximate to marine life
            return ("⚠️   PROCEED WITH CARE — Mixed Scene", c["uncertain"])

        if r.uncertain_count > 0 and r.trash_count == 0 and r.marine_count == 0:
            return ("⏳  SCANNING — Uncertain Objects", c["uncertain"])

        if r.trash_count == 0 and r.marine_count == 0:
            return ("✅  CLEAR — No Targets Detected", c["accent"])

        return ("⏳  SCANNING…", c["hud_text"])
