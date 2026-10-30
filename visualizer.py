"""
visualizer.py — OpenCV rendering engine for Ocean Guardian.

Draws everything the judges see:
  - Bounding boxes styled per category (marine / trash / uncertain / blocked)
  - Labels with confidence percentages
  - Category icons (emoji-style prefix)
  - Right-side HUD panel with scene summary, counts, risk level, robot command
  - Animated "scanning" line to signal real-time processing
  - Frame timestamp

Design philosophy: dark ocean-terminal aesthetic — deep navy panels,
teal accents, high-contrast labels. Feels like a real submersible HUD.
"""

import cv2
import numpy as np
import time
from typing import List, Tuple

from detector import Detection
from decision_engine import SceneReport
from config import Config, COLORS


class Visualizer:
    """
    All drawing logic is here. main.py just calls viz.render(frame, ...).
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._frame_count = 0
        self._start_time = time.time()

        # OpenCV font
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.font_small = cv2.FONT_HERSHEY_SIMPLEX

    # ── Public API ────────────────────────────────────────────────────────────

    def render(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        report: SceneReport,
    ) -> np.ndarray:
        """
        Compose the full output frame:
          1. Draw bounding boxes + labels on the video feed
          2. Attach side HUD panel
        Returns the composited BGR frame.
        """
        self._frame_count += 1

        # Work on a copy so original frame isn't mutated
        canvas = frame.copy()

        # Step 1 — bounding boxes
        self._draw_detections(canvas, detections, report)

        # Step 2 — scan line overlay (cosmetic real-time effect)
        self._draw_scan_line(canvas)

        # Step 3 — top-left watermark / system ID
        self._draw_watermark(canvas)

        # Step 4 — compose HUD panel to the right of the video
        output = self._attach_hud_panel(canvas, report)

        return output

    # ── Detection boxes ───────────────────────────────────────────────────────

    def _draw_detections(
        self,
        canvas: np.ndarray,
        detections: List[Detection],
        report: SceneReport,
    ):
        blocked_set = set(id(d) for d in report.blocked)

        for det in detections:
            is_blocked = id(det) in blocked_set
            self._draw_box(canvas, det, is_blocked)

    def _draw_box(
        self,
        canvas: np.ndarray,
        det: Detection,
        is_blocked: bool,
    ):
        x1, y1, x2, y2 = det.bbox
        cfg = self.cfg

        # ── Pick color ────────────────────────────────────────────────────────
        if is_blocked:
            color = COLORS["critical"]
            tag = "⛔ BLOCKED"
        elif det.category == "marine":
            color = COLORS["marine"]
            tag = "🐠 MARINE LIFE"
        elif det.category == "trash":
            color = COLORS["trash"]
            tag = "♻️  TRASH"
        else:
            color = COLORS["uncertain"]
            tag = "⚠️  UNCERTAIN"

        # ── Bounding box ──────────────────────────────────────────────────────
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, cfg.BOX_THICKNESS)

        # Corner accent lines (makes box look less generic)
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        thick = cfg.BOX_THICKNESS + 1
        # Top-left
        cv2.line(canvas, (x1, y1), (x1 + corner_len, y1), color, thick)
        cv2.line(canvas, (x1, y1), (x1, y1 + corner_len), color, thick)
        # Bottom-right
        cv2.line(canvas, (x2, y2), (x2 - corner_len, y2), color, thick)
        cv2.line(canvas, (x2, y2), (x2, y2 - corner_len), color, thick)

        # ── Label background pill ─────────────────────────────────────────────
        label_text = f"{det.label}  {det.confidence:.0%}"
        (lw, lh), _ = cv2.getTextSize(
            label_text, self.font_small, cfg.FONT_SCALE_LABEL, 1
        )
        pad = 5
        # Place above box, clamp to canvas top
        label_y = max(y1 - lh - pad * 2, lh + pad)
        cv2.rectangle(
            canvas,
            (x1, label_y - lh - pad),
            (x1 + lw + pad * 2, label_y + pad),
            color,
            -1,  # filled
        )
        cv2.putText(
            canvas, label_text,
            (x1 + pad, label_y),
            self.font_small, cfg.FONT_SCALE_LABEL,
            (10, 10, 10),  # dark text on colored bg
            1, cv2.LINE_AA,
        )

        # ── Action tag below box ──────────────────────────────────────────────
        (tw, th), _ = cv2.getTextSize(
            tag, self.font_small, cfg.FONT_SCALE_LABEL * 0.9, 1
        )
        tag_y = y2 + th + pad * 2
        # Semi-transparent dark background for readability
        overlay = canvas.copy()
        cv2.rectangle(
            overlay,
            (x1, y2 + pad),
            (x1 + tw + pad * 2, tag_y + pad),
            COLORS["hud_bg"],
            -1,
        )
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
        cv2.putText(
            canvas, tag,
            (x1 + pad, tag_y),
            self.font_small, cfg.FONT_SCALE_LABEL * 0.9,
            color, 1, cv2.LINE_AA,
        )

        # ── Risk score dot ────────────────────────────────────────────────────
        # Small filled circle at bottom-right of box indicating risk_score
        risk_radius = int(det.risk_score * 12) + 3
        cx, cy = det.center
        cv2.circle(canvas, (x2 - 8, y2 - 8), risk_radius, color, -1)

    # ── HUD side panel ────────────────────────────────────────────────────────

    def _attach_hud_panel(
        self,
        video_frame: np.ndarray,
        report: SceneReport,
    ) -> np.ndarray:
        """Create a dark side panel and concatenate it to the right of the frame."""
        h, w = video_frame.shape[:2]
        panel_w = self.cfg.PANEL_WIDTH

        # Dark navy background
        panel = np.full((h, panel_w, 3), COLORS["panel_bg"], dtype=np.uint8)

        # ── Draw all HUD sections ─────────────────────────────────────────────
        y = self._hud_header(panel, y=18)
        y = self._hud_divider(panel, y)
        y = self._hud_command(panel, report, y)
        y = self._hud_divider(panel, y)
        y = self._hud_counts(panel, report, y)
        y = self._hud_divider(panel, y)
        y = self._hud_risk_bar(panel, report, y)
        y = self._hud_divider(panel, y)
        y = self._hud_detection_list(panel, report, y)
        self._hud_footer(panel)

        return np.hstack([video_frame, panel])

    def _hud_header(self, panel, y):
        self._put(panel, "OCEAN GUARDIAN", 10, y, COLORS["accent"],
                  scale=0.65, thickness=2)
        self._put(panel, "AI Cleanup System  v1.0", 10, y + 20,
                  COLORS["hud_text"], scale=0.40)
        return y + 44

    def _hud_command(self, panel, report: SceneReport, y):
        self._put(panel, "ROBOT COMMAND", 10, y, COLORS["hud_text"],
                  scale=0.40)
        y += 18
        # Command in big bold text, colored by urgency
        cmd = report.robot_command
        # Word-wrap long commands at 28 chars
        lines = self._wrap(cmd, 28)
        for line in lines:
            self._put(panel, line, 10, y, report.command_color,
                      scale=0.52, thickness=2)
            y += 22
        return y + 4

    def _hud_counts(self, panel, report: SceneReport, y):
        self._put(panel, "DETECTIONS", 10, y, COLORS["hud_text"], scale=0.40)
        y += 18

        items = [
            ("🐠 Marine Life",  report.marine_count,   COLORS["marine"]),
            ("♻️  Trash",        report.trash_count,    COLORS["trash"]),
            ("⚠️  Uncertain",    report.uncertain_count, COLORS["uncertain"]),
        ]
        for label, count, color in items:
            # Mini bar
            bar_len = min(count * 18, self.cfg.PANEL_WIDTH - 70)
            if bar_len > 0:
                cv2.rectangle(panel, (10, y - 8), (10 + bar_len, y + 2), color, -1)
            self._put(panel, f"{label}: {count}", 10, y + 12, color, scale=0.44)
            y += 26

        # Collectible vs blocked sub-line
        if report.trash_count > 0:
            self._put(
                panel,
                f"  ✓ Collectible: {len(report.collectible)}  "
                f"✗ Blocked: {len(report.blocked)}",
                10, y, COLORS["hud_text"], scale=0.38,
            )
            y += 18
        return y + 4

    def _hud_risk_bar(self, panel, report: SceneReport, y):
        self._put(panel, "SCENE RISK", 10, y, COLORS["hud_text"], scale=0.40)
        y += 16

        risk_map = {
            "CLEAR":    (0.0,  COLORS["marine"]),
            "LOW":      (0.2,  COLORS["marine"]),
            "MEDIUM":   (0.5,  COLORS["uncertain"]),
            "HIGH":     (0.75, COLORS["trash"]),
            "CRITICAL": (1.0,  COLORS["critical"]),
        }
        frac, risk_color = risk_map.get(report.scene_risk, (0.5, COLORS["uncertain"]))
        bar_total = self.cfg.PANEL_WIDTH - 24
        bar_fill  = int(frac * bar_total)

        # Background track
        cv2.rectangle(panel, (12, y), (12 + bar_total, y + 12),
                      (40, 50, 55), -1)
        # Fill
        if bar_fill > 0:
            cv2.rectangle(panel, (12, y), (12 + bar_fill, y + 12),
                          risk_color, -1)

        y += 16
        self._put(panel, f"  {report.scene_risk}", 10, y, risk_color,
                  scale=0.48, thickness=2)
        return y + 20

    def _hud_detection_list(self, panel, report: SceneReport, y):
        """List the top 5 individual detections."""
        self._put(panel, "ACTIVE TARGETS", 10, y, COLORS["hud_text"], scale=0.40)
        y += 18

        all_dets = report.collectible + report.blocked
        # Sort by risk desc
        all_dets.sort(key=lambda d: d.risk_score, reverse=True)
        blocked_ids = set(id(d) for d in report.blocked)

        for i, det in enumerate(all_dets[:5]):
            status = "BLOCKED" if id(det) in blocked_ids else "TARGET "
            color = COLORS["critical"] if id(det) in blocked_ids else COLORS["trash"]
            line = f"  {i+1}. [{status}] {det.label[:12]:<12} {det.confidence:.0%}"
            self._put(panel, line, 8, y, color, scale=0.38)
            y += 16

        if not all_dets:
            self._put(panel, "  — No active targets —", 10, y,
                      COLORS["hud_text"], scale=0.38)
            y += 16

        return y + 4

    def _hud_footer(self, panel):
        """Draw session timer and FPS hint at the bottom of the panel."""
        h = panel.shape[0]
        elapsed = int(time.time() - self._start_time)
        m, s = divmod(elapsed, 60)
        timer_str = f"Session: {m:02d}:{s:02d}"
        self._put(panel, timer_str, 10, h - 24,
                  COLORS["hud_text"], scale=0.38)
        self._put(panel, "Press Q to quit  |  S = screenshot",
                  10, h - 10, (80, 90, 85), scale=0.33)

    # ── Cosmetic overlays ─────────────────────────────────────────────────────

    def _draw_scan_line(self, canvas: np.ndarray):
        """A moving horizontal scan line — makes the demo feel alive."""
        h, w = canvas.shape[:2]
        period = 90  # frames per full sweep
        y = int((self._frame_count % period) / period * h)
        overlay = canvas.copy()
        cv2.line(overlay, (0, y), (w, y), COLORS["accent"], 1)
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

    def _draw_watermark(self, canvas: np.ndarray):
        """Small system label top-left corner."""
        self._put(canvas, "OCEAN GUARDIAN  |  LIVE",
                  8, 20, COLORS["accent"], scale=0.50, thickness=1)

    # ── Utility helpers ───────────────────────────────────────────────────────

    def _hud_divider(self, panel, y):
        w = panel.shape[1]
        cv2.line(panel, (8, y), (w - 8, y), (45, 55, 60), 1)
        return y + 10

    def _put(
        self,
        img: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: Tuple[int, int, int],
        scale: float = 0.5,
        thickness: int = 1,
    ):
        """Convenience wrapper for cv2.putText with AA."""
        cv2.putText(
            img, text, (x, y),
            self.font_small, scale, color, thickness, cv2.LINE_AA,
        )

    @staticmethod
    def _wrap(text: str, width: int) -> List[str]:
        """Very simple word-wrap by character count."""
        words = text.split()
        lines, current = [], ""
        for w in words:
            if len(current) + len(w) + 1 <= width:
                current = (current + " " + w).strip()
            else:
                if current:
                    lines.append(current)
                current = w
        if current:
            lines.append(current)
        return lines or [text]
