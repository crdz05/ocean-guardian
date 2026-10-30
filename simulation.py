"""
simulation.py — Top-down pygame simulation for the Ocean Guardian demo.

A collector robot patrols a patch of ocean, drives to the nearest trash and
picks it up — unless marine life is inside the safety radius. Then it HOLDs
for Config.SAFETY_WAIT_SECONDS, shows a countdown, and re-checks: collect if
the animal left, otherwise wait again or switch to another (safe) trash item.

Two inputs drive the robot at the same time:
  1. The simulated world, evaluated by the real DecisionEngine. Sim objects
     are turned into Detection objects, so the same proximity rule that blocks
     trash near marine life in the camera demo blocks it here
     (margin = Config.SIM_SAFETY_RADIUS).
  2. A "camera" signal: the webcam + YOLO detector running in a background
     thread, or the F / T keyboard overrides when the camera or torch is
     unavailable. Marine life on camera → robot HOLDs. Trash on camera →
     COLLECT allowed (a new trash item is dropped near the robot).

Run via main.py:
    python main.py --sim               # simulation + webcam
    python main.py --sim --no-camera   # simulation only (F/T keys as camera)
"""

import dataclasses
import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pygame

from config import Config
from detector import Detection          # plain dataclass; no torch import
from decision_engine import DecisionEngine, SceneReport


# ── Palette (RGB for pygame) ─────────────────────────────────────────────────

C = {
    "deep":      (5, 32, 58),
    "shallow":   (12, 84, 118),
    "wave":      (28, 108, 146),
    "marine":    (0, 210, 160),
    "trash":     (235, 80, 60),
    "uncertain": (255, 170, 0),
    "critical":  (255, 45, 45),
    "accent":    (0, 220, 170),
    "moving":    (90, 170, 255),
    "text":      (225, 240, 235),
    "dim":       (120, 140, 140),
    "panel":     (14, 20, 28),
    "panel_2":   (24, 34, 44),
    "hull":      (250, 190, 40),
}

STATE_COLORS = {
    "SCANNING":   C["accent"],
    "MOVING":     C["moving"],
    "HOLD":       C["critical"],
    "COLLECTING": C["uncertain"],
}

COMMAND_COLORS = {
    "HOLD":              C["critical"],
    "OBSERVE":           C["marine"],
    "PRIORITY CLEANUP":  C["trash"],
    "COLLECT":           C["trash"],
    "PROCEED WITH CARE": C["uncertain"],
    "SCANNING":          C["uncertain"],
    "CLEAR":             C["accent"],
}


def clean_text(text: str) -> str:
    """Strip emoji / non-ASCII from engine strings (pygame fonts lack them)."""
    text = text.replace("—", "-").replace("…", "...")
    text = "".join(ch for ch in text if ord(ch) < 128)
    return " ".join(text.split())


# ── Sprites (simple shapes, drawn once, facing right) ────────────────────────

def _fish_sprite(size: int, body, fin) -> pygame.Surface:
    w, h = int(size * 2.2), int(size * 1.2)
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.polygon(s, fin, [(0, 0), (int(w * 0.3), h // 2), (0, h)])
    pygame.draw.ellipse(s, body, (int(w * 0.18), int(h * 0.1), int(w * 0.8), int(h * 0.8)))
    pygame.draw.polygon(s, fin, [(int(w * 0.48), int(h * 0.18)),
                                 (int(w * 0.6), 0), (int(w * 0.7), int(h * 0.2))])
    ex, ey = int(w * 0.82), int(h * 0.42)
    pygame.draw.circle(s, (255, 255, 255), (ex, ey), max(2, size // 5))
    pygame.draw.circle(s, (10, 10, 10), (ex + 1, ey), max(1, size // 10))
    return s


def _turtle_sprite(size: int) -> pygame.Surface:
    w = h = int(size * 2.8)
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    cx, cy = w // 2, h // 2
    flipper = (70, 150, 95)
    for dx, dy in ((0.45, -0.78), (0.45, 0.78), (-0.55, -0.62), (-0.55, 0.62)):
        fw, fh = size * 0.75, size * 0.42
        pygame.draw.ellipse(s, flipper, (cx + dx * size - fw / 2, cy + dy * size - fh / 2, fw, fh))
    pygame.draw.circle(s, flipper, (int(cx + size * 0.98), cy), int(size * 0.32))
    pygame.draw.ellipse(s, (40, 105, 58), (cx - size * 0.82, cy - size * 0.66, size * 1.64, size * 1.32))
    pygame.draw.ellipse(s, (72, 145, 80), (cx - size * 0.55, cy - size * 0.42, size * 1.1, size * 0.84), 2)
    for ox, oy in ((-0.3, 0), (0.3, 0), (0, -0.38), (0, 0.38)):
        pygame.draw.circle(s, (72, 145, 80), (int(cx + ox * size), int(cy + oy * size)), max(2, size // 7))
    return s


def _bottle_sprite(size: int) -> pygame.Surface:
    w, h = int(size * 2.4), int(size * 1.0)
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(s, (160, 215, 240, 210), (0, int(h * 0.12), int(w * 0.68), int(h * 0.76)),
                     border_radius=max(3, size // 3))
    pygame.draw.rect(s, (160, 215, 240, 210), (int(w * 0.66), int(h * 0.32), int(w * 0.2), int(h * 0.36)))
    pygame.draw.rect(s, (30, 90, 210), (int(w * 0.85), int(h * 0.26), int(w * 0.15), int(h * 0.48)))
    pygame.draw.rect(s, (235, 245, 250, 230), (int(w * 0.12), int(h * 0.3), int(w * 0.38), int(h * 0.4)))
    return s


def _bag_sprite(size: int) -> pygame.Surface:
    w = h = int(size * 2.4)
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    pts = [(0.2, 0.3), (0.45, 0.22), (0.8, 0.28), (0.92, 0.6),
           (0.75, 0.92), (0.4, 0.95), (0.1, 0.75)]
    pygame.draw.polygon(s, (235, 238, 245, 170), [(x * w, y * h) for x, y in pts])
    pygame.draw.polygon(s, (255, 255, 255, 220), [(x * w, y * h) for x, y in pts], 2)
    pygame.draw.arc(s, (255, 255, 255, 220), (w * 0.22, 0, w * 0.22, h * 0.45), 0, math.pi, 2)
    pygame.draw.arc(s, (255, 255, 255, 220), (w * 0.55, h * 0.02, w * 0.22, h * 0.45), 0, math.pi, 2)
    return s


def _net_sprite(size: int) -> pygame.Surface:
    w = h = int(size * 2.4)
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    rope = (175, 145, 90)
    step = max(6, w // 6)
    for i in range(-w, w * 2, step):
        pygame.draw.line(s, rope, (i, 0), (i + h, h), 2)
        pygame.draw.line(s, rope, (i, h), (i + h, 0), 2)
    mask = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.ellipse(mask, (255, 255, 255, 255), (0, 0, w, h))
    s.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    pygame.draw.ellipse(s, rope, (0, 0, w, h), 2)
    for ang in (0.3, 2.2, 4.1):
        pygame.draw.circle(s, (255, 120, 30),
                           (int(w / 2 + math.cos(ang) * w * 0.47), int(h / 2 + math.sin(ang) * h * 0.47)), 4)
    return s


# ── World entities ───────────────────────────────────────────────────────────

@dataclass(eq=False)          # eq=False → hashable by identity (used in sets)
class Trash:
    kind: str                  # "bottle" | "bag" | "net"
    x: float
    y: float
    size: float
    angle: float
    sprite: pygame.Surface
    phase: float = field(default_factory=lambda: random.uniform(0, math.tau))

    def bbox(self) -> Tuple[int, int, int, int]:
        r = self.size
        return (int(self.x - r), int(self.y - r), int(self.x + r), int(self.y + r))


@dataclass(eq=False)
class Animal:
    kind: str                  # "fish" | "turtle" | "jellyfish"
    x: float
    y: float
    heading: float
    speed: float
    size: float
    born: float
    life: float                # seconds before it swims away
    sprite: Optional[pygame.Surface] = None
    phase: float = field(default_factory=lambda: random.uniform(0, math.tau))
    leaving: bool = False

    def bbox(self) -> Tuple[int, int, int, int]:
        r = self.size
        return (int(self.x - r), int(self.y - r), int(self.x + r), int(self.y + r))


# ── Camera bridge (webcam thread + keyboard overrides) ───────────────────────

class CameraBridge:
    """
    Shared state between the camera thread, the F/T/C key overrides and the
    simulation. The webcam thread runs the existing detector, DecisionEngine
    and Visualizer box drawing; the simulation only reads the latest
    SceneReport and frame.
    """

    # Synthetic detections used by the F / T demo overrides. They go through
    # a real DecisionEngine so the panel shows the engine's own command.
    _OVERRIDES = {
        "marine": ("F: fish detected",
                   lambda: [Detection("fish", "marine", 0.92, (220, 140, 380, 260),
                                      0.0, "AVOID - Marine Life")]),
        "trash":  ("T: trash detected",
                   lambda: [Detection("plastic_bottle", "trash", 0.88, (720, 300, 820, 470),
                                      0.97, "COLLECT - Confirmed Trash")]),
    }

    def __init__(self, cfg: Config, use_camera: bool):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.status = "Starting camera..." if use_camera else "Camera off (--no-camera)"
        self.live = False
        self.frame_id = 0
        self._frame = None                      # RGB numpy array
        self._cam_report: Optional[SceneReport] = None
        self._cam_marine_desc = ""              # e.g. "fish 72% (COCO: bird)"
        self._override_label: Optional[str] = None
        self._override_report: Optional[SceneReport] = None
        self._override_engine = DecisionEngine(cfg)
        self._thread: Optional[threading.Thread] = None
        if use_camera:
            self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
            self._thread.start()

    # ── Overrides ─────────────────────────────────────────────────────────────

    def set_override(self, kind: str):
        label, make_dets = self._OVERRIDES[kind]
        report = self._override_engine.evaluate(make_dets())
        with self._lock:
            self._override_label = label
            self._override_report = report
        print(f"[DEMO] Override -> {label}  |  engine: {clean_text(report.robot_command)}")

    def clear_override(self):
        with self._lock:
            self._override_label = None
            self._override_report = None
        print("[DEMO] Override cleared")

    # ── Read side (simulation thread) ─────────────────────────────────────────

    def signal(self) -> Tuple[str, Optional[SceneReport]]:
        """(source, report): key override wins over the live camera."""
        with self._lock:
            if self._override_report is not None:
                return self._override_label, self._override_report
            if self.live and self._cam_report is not None:
                return "CAMERA (YOLOv8)", self._cam_report
        return "none", None

    def marine_desc(self) -> str:
        """What the camera/override sees as marine life, for logs and HUD."""
        with self._lock:
            if self._override_report is not None:
                return "fish 92% (key F)" if self._override_report.marine_count else ""
            return self._cam_marine_desc if self.live else ""

    def latest_frame(self):
        with self._lock:
            return self.frame_id, self._frame

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ── Camera thread ─────────────────────────────────────────────────────────

    def _run(self):
        self.status = "Loading YOLOv8..."
        try:
            import cv2
            from detector import OceanGuardianDetector
            from visualizer import Visualizer
            detector = OceanGuardianDetector(self.cfg)
        except Exception as exc:   # ImportError, OSError (WinError 4551), ...
            print("\n[WARN] -------------------------------------------------------")
            print(f"[WARN] Camera detector unavailable: {type(exc).__name__}: {exc}")
            print("[WARN] torch / ultralytics could not be loaded. The simulation keeps")
            print("[WARN] running: use F (fish detected), T (trash detected), C (clear)")
            print("[WARN] as the camera input.")
            print("[WARN] -------------------------------------------------------\n")
            self.status = "No detector - use F/T keys"
            return

        engine = DecisionEngine(self.cfg)
        viz = Visualizer(self.cfg)
        cap = cv2.VideoCapture(self.cfg.CAMERA_INDEX)
        if not cap.isOpened():
            print(f"[WARN] Could not open camera {self.cfg.CAMERA_INDEX}. Use F/T keys.")
            self.status = "No webcam - use F/T keys"
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.status = "LIVE - YOLOv8"
        self.live = True
        print("[INFO] Camera thread running.")

        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.status = "Camera read failed - use F/T keys"
                    break
                detections = detector.detect(frame)
                report = engine.evaluate(detections)
                canvas = frame.copy()
                viz._draw_detections(canvas, detections, report)   # same boxes as camera mode
                viz._draw_watermark(canvas)
                rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                marine = [d for d in detections if d.category == "marine"]
                desc = ""
                if marine:
                    m = marine[0]
                    desc = f"{m.label} {m.confidence:.0%}"
                    if m.raw_label and m.raw_label != m.label:
                        desc += f" (COCO: {m.raw_label})"
                with self._lock:
                    self._frame = rgb
                    self._cam_report = report
                    self._cam_marine_desc = desc
                    self.frame_id += 1
        except Exception as exc:
            print(f"[WARN] Camera thread stopped: {type(exc).__name__}: {exc}")
            self.status = "Camera error - use F/T keys"
        finally:
            self.live = False
            cap.release()


# ── Simulation ───────────────────────────────────────────────────────────────

class Simulation:
    PANEL_W = 380
    MARGIN = 50
    CAMERA_TRASH_COOLDOWN = 3.0

    def __init__(self, cfg: Config, use_camera: bool = True, duration: Optional[float] = None):
        pygame.init()
        self.cfg = cfg
        self.W, self.H = cfg.SIM_WIDTH, cfg.SIM_HEIGHT
        self.ocean_w = self.W - self.PANEL_W
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Ocean Guardian - Simulation")
        self.clock = pygame.time.Clock()
        self.duration = duration

        mono = "consolas,menlo,dejavusansmono,couriernew"
        self.f_small = pygame.font.SysFont(mono, 14)
        self.f_med = pygame.font.SysFont(mono, 17, bold=True)
        self.f_big = pygame.font.SysFont(mono, 26, bold=True)

        # The world is judged by the real DecisionEngine, with the sim's
        # safety radius as its proximity margin.
        world_cfg = dataclasses.replace(cfg, PROXIMITY_SAFETY_MARGIN_PX=cfg.SIM_SAFETY_RADIUS)
        self.world_engine = DecisionEngine(world_cfg)

        self.bridge = CameraBridge(cfg, use_camera)
        self._bg = self._make_background()
        self._overlay = pygame.Surface((self.ocean_w, self.H), pygame.SRCALPHA)
        self._cam_surface = None
        self._cam_frame_id = -1
        self.reset()

    def reset(self):
        self.t = 0.0
        self.rx, self.ry = self.ocean_w / 2, self.H / 2
        self.heading = 0.0
        self.state = "SCANNING"
        self.target: Optional[Trash] = None
        self.wander: Optional[Tuple[float, float]] = None
        self.trash: List[Trash] = []
        self.animals: List[Animal] = []
        self.blocked: set = set()
        self.world_report = SceneReport()
        self.collected = 0
        self.protected = 0
        self.hold_timer = 0.0
        self.hold_reason = ""
        self.collect_timer = 0.0
        self.effects: List[Tuple[float, float, float]] = []   # (x, y, start_t)
        self.next_trash_t = 2.0
        self.next_animal_t = 4.0
        self.cam_source, self.cam_report = "none", None
        self.cam_marine = self.cam_trash = False
        self._prev_cam_trash = False
        self._last_cam_spawn = -99.0
        for _ in range(4):
            self._spawn_trash()
        self._spawn_animal(near_trash=True)
        self._spawn_animal(near_trash=False)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        print("[INFO] Simulation running. F = fish, T = trash, C = clear, "
              "S = screenshot, Q/ESC = quit.")
        running = True
        try:
            while running:
                dt = min(self.clock.tick(self.cfg.SIM_FPS) / 1000.0, 0.05)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        running = self._on_key(event.key)
                self.update(dt)
                self.draw()
                pygame.display.flip()
                if self.duration is not None and self.t >= self.duration:
                    running = False
        finally:
            self.bridge.stop()
            pygame.quit()
            self.print_summary()

    def _on_key(self, key) -> bool:
        if key in (pygame.K_q, pygame.K_ESCAPE):
            return False
        if key == pygame.K_f:
            self.bridge.set_override("marine")
        elif key == pygame.K_t:
            self.bridge.set_override("trash")
        elif key == pygame.K_c:
            self.bridge.clear_override()
        elif key == pygame.K_s:
            fname = f"sim_screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            pygame.image.save(self.screen, fname)
            print(f"[INFO] Screenshot saved -> {fname}")
        return True

    def print_summary(self):
        m, s = divmod(int(self.t), 60)
        print(f"\n[INFO] Simulation ended after {m:02d}:{s:02d} | "
              f"trash collected: {self.collected} | animals protected (holds): {self.protected}")

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, dt: float):
        self.t += dt
        self._spawn_tick()
        self._move_animals(dt)
        self._evaluate_world()

        self.cam_source, self.cam_report = self.bridge.signal()
        r = self.cam_report
        self.cam_marine = bool(r and r.marine_count > 0)
        self.cam_trash = bool(r and r.command_key in ("COLLECT", "PRIORITY CLEANUP"))
        if (self.cam_trash and not self._prev_cam_trash
                and self.t - self._last_cam_spawn > self.CAMERA_TRASH_COOLDOWN
                and len(self.trash) < self.cfg.SIM_MAX_TRASH + 3):
            self._spawn_trash(near_robot=True)
            self._last_cam_spawn = self.t
            print("[SIM] Camera saw trash -> new collection target dropped near robot")
        self._prev_cam_trash = self.cam_trash

        self._step_robot(dt)
        self.effects = [e for e in self.effects if self.t - e[2] < 0.8]

    def _evaluate_world(self):
        """Feed the sim world to the DecisionEngine as Detection objects."""
        dets, owner = [], {}
        for t in self.trash:
            d = Detection(t.kind, "trash", 1.0, t.bbox(), 1.0, "COLLECT")
            dets.append(d)
            owner[id(d)] = t
        for a in self.animals:
            dets.append(Detection(a.kind, "marine", 1.0, a.bbox(), 0.0, "AVOID"))
        self.world_report = self.world_engine.evaluate(dets)
        self.blocked = {owner[id(d)] for d in self.world_report.blocked}

    def _step_robot(self, dt: float):
        radius = self.cfg.SIM_SAFETY_RADIUS
        near = self._animal_near(self.rx, self.ry, radius)
        if self.target is not None and self.target not in self.trash:
            self.target = None

        # Safety first: camera marine life or an animal at the robot → HOLD.
        if self.state != "HOLD":
            if self.cam_marine:
                self._enter_hold(f"Camera: {self.bridge.marine_desc() or 'marine life'}")
                return
            if near is not None:
                self._enter_hold(f"{near.kind.title()} inside robot safety radius")
                return

        if self.state == "HOLD":
            self.hold_timer -= dt
            if self.hold_timer <= 0:
                self._resolve_hold(near)
            return

        if self.state == "SCANNING":
            self.target = self._pick_target()
            if self.target is None:
                self._patrol(dt)
                return
            self.state = "MOVING"

        tgt = self.target
        if tgt is None:
            self.state = "SCANNING"
            return
        dist = math.hypot(tgt.x - self.rx, tgt.y - self.ry)

        if self.state == "MOVING":
            if tgt in self.blocked:
                alt = self._nearest_trash(safe_only=True)
                if alt is not None:
                    self.target = alt
                    return
                if dist < radius * 2.2:
                    self._enter_hold(f"Marine life near the {tgt.kind}")
                    return
            if dist <= tgt.size + 14:
                self.state = "COLLECTING"
                self.collect_timer = self.cfg.SIM_COLLECT_SECONDS
                return
            self._drive_toward(tgt.x, tgt.y, dt)

        elif self.state == "COLLECTING":
            if tgt in self.blocked:
                self._enter_hold(f"Marine life near the {tgt.kind}")
                return
            self.collect_timer -= dt
            if self.collect_timer <= 0:
                self.trash.remove(tgt)
                self.collected += 1
                self.effects.append((tgt.x, tgt.y, self.t))
                print(f"[SIM] Collected {tgt.kind}  (total {self.collected})")
                self.target = None
                self.state = "SCANNING"

    def _enter_hold(self, reason: str):
        self.state = "HOLD"
        self.hold_timer = self.cfg.SAFETY_WAIT_SECONDS
        self.hold_reason = reason
        self.protected += 1
        print(f"[SIM] HOLD {self.cfg.SAFETY_WAIT_SECONDS:.0f}s - {reason}")

    def _resolve_hold(self, near: Optional[Animal]):
        """Countdown finished: collect if clear, else wait again or retarget."""
        if self.cam_marine:
            self._enter_hold(f"Camera: {self.bridge.marine_desc() or 'marine life'} still in view")
            return
        if near is not None:
            self._enter_hold(f"{near.kind.title()} still near the robot")
            return
        tgt = self.target if self.target in self.trash else None
        if tgt is not None and tgt not in self.blocked:
            self.state = "MOVING"
            return
        alt = self._nearest_trash(safe_only=True)
        if alt is not None:
            self.target = alt
            self.state = "MOVING"
            print(f"[SIM] Switching to another {alt.kind} (safe)")
            return
        if tgt is not None:
            self._enter_hold(f"Marine life still near the {tgt.kind}")
            return
        self.state = "SCANNING"

    def _pick_target(self) -> Optional[Trash]:
        # Prefer safe trash; otherwise approach a blocked one (and HOLD there).
        return self._nearest_trash(safe_only=True) or self._nearest_trash(safe_only=False)

    def _nearest_trash(self, safe_only: bool) -> Optional[Trash]:
        pool = [t for t in self.trash if not (safe_only and t in self.blocked)]
        if not pool:
            return None
        return min(pool, key=lambda t: (t.x - self.rx) ** 2 + (t.y - self.ry) ** 2)

    def _animal_near(self, x: float, y: float, radius: float) -> Optional[Animal]:
        for a in self.animals:
            if math.hypot(a.x - x, a.y - y) < radius:
                return a
        return None

    def _drive_toward(self, tx: float, ty: float, dt: float, speed_factor: float = 1.0):
        desired = math.atan2(ty - self.ry, tx - self.rx)
        diff = (desired - self.heading + math.pi) % math.tau - math.pi
        turn = 4.0 * dt
        self.heading += max(-turn, min(turn, diff))
        speed = self.cfg.SIM_ROBOT_SPEED * speed_factor * max(0.25, math.cos(diff))
        self.rx += math.cos(self.heading) * speed * dt
        self.ry += math.sin(self.heading) * speed * dt
        self.rx = max(20, min(self.ocean_w - 20, self.rx))
        self.ry = max(20, min(self.H - 20, self.ry))

    def _patrol(self, dt: float):
        if self.wander is None or math.hypot(self.wander[0] - self.rx, self.wander[1] - self.ry) < 20:
            self.wander = (random.uniform(self.MARGIN, self.ocean_w - self.MARGIN),
                           random.uniform(self.MARGIN, self.H - self.MARGIN))
        self._drive_toward(*self.wander, dt, speed_factor=0.45)

    # ── Spawning / animals ────────────────────────────────────────────────────

    def _spawn_tick(self):
        if self.t >= self.next_trash_t:
            if len(self.trash) < self.cfg.SIM_MAX_TRASH:
                self._spawn_trash()
            self.next_trash_t = self.t + random.uniform(2.5, 4.5)
        if self.t >= self.next_animal_t:
            if len(self.animals) < self.cfg.SIM_MAX_ANIMALS:
                self._spawn_animal(near_trash=random.random() < 0.5)
            self.next_animal_t = self.t + random.uniform(4.0, 8.0)

    def _spawn_trash(self, near_robot: bool = False):
        kind = random.choice(("bottle", "bottle", "bag", "net"))
        size = {"bottle": 13, "bag": 14, "net": 20}[kind]
        m = self.MARGIN
        for _ in range(30):
            if near_robot:
                ang, d = random.uniform(0, math.tau), random.uniform(140, 220)
                x, y = self.rx + math.cos(ang) * d, self.ry + math.sin(ang) * d
            else:
                x, y = random.uniform(m, self.ocean_w - m), random.uniform(m, self.H - m)
            inside = m <= x <= self.ocean_w - m and m <= y <= self.H - m
            if inside and (near_robot or math.hypot(x - self.rx, y - self.ry) > 120):
                break
        else:
            return
        sprite = {"bottle": _bottle_sprite, "bag": _bag_sprite, "net": _net_sprite}[kind](size)
        self.trash.append(Trash(kind, x, y, size, random.uniform(0, 360), sprite))

    def _spawn_animal(self, near_trash: bool):
        kind = random.choice(("fish", "fish", "turtle", "jellyfish"))
        size = {"fish": 14, "turtle": 17, "jellyfish": 14}[kind]
        speed = self.cfg.SIM_ANIMAL_SPEED * {"fish": 1.25, "turtle": 0.8, "jellyfish": 0.5}[kind]
        m = self.MARGIN
        x = y = None
        if near_trash and self.trash:
            t = random.choice(self.trash)
            ang = random.uniform(0, math.tau)
            d = random.uniform(150, 210)
            cx, cy = t.x + math.cos(ang) * d, t.y + math.sin(ang) * d
            if m <= cx <= self.ocean_w - m and m <= cy <= self.H - m:
                x, y, heading = cx, cy, math.atan2(t.y - cy, t.x - cx)   # swims toward it
        if x is None:
            side = random.randrange(4)
            if side == 0:
                x, y, heading = -20, random.uniform(m, self.H - m), 0.0
            elif side == 1:
                x, y, heading = self.ocean_w + 20, random.uniform(m, self.H - m), math.pi
            elif side == 2:
                x, y, heading = random.uniform(m, self.ocean_w - m), -20, math.pi / 2
            else:
                x, y, heading = random.uniform(m, self.ocean_w - m), self.H + 20, -math.pi / 2
            heading += random.uniform(-0.4, 0.4)
        sprite = None
        if kind == "fish":
            body, fin = random.choice((((255, 150, 40), (255, 110, 20)),
                                       ((255, 220, 60), (240, 170, 30)),
                                       ((120, 200, 255), (60, 140, 230))))
            sprite = _fish_sprite(size, body, fin)
        elif kind == "turtle":
            sprite = _turtle_sprite(size)
        self.animals.append(Animal(kind, x, y, heading, speed, size, self.t,
                                   random.uniform(14, 26), sprite))

    def _move_animals(self, dt: float):
        m = 30
        keep = []
        for a in self.animals:
            if not a.leaving and self.t - a.born > a.life:
                a.leaving = True
            a.heading += random.uniform(-1.2, 1.2) * dt
            a.x += math.cos(a.heading) * a.speed * dt
            a.y += math.sin(a.heading) * a.speed * dt
            if not a.leaving and self.t - a.born > 2.0:   # bounce inside the ocean
                if a.x < m or a.x > self.ocean_w - m:
                    a.heading = math.pi - a.heading
                    a.x = max(m, min(self.ocean_w - m, a.x))
                if a.y < m or a.y > self.H - m:
                    a.heading = -a.heading
                    a.y = max(m, min(self.H - m, a.y))
            out = a.x < -60 or a.x > self.ocean_w + 60 or a.y < -60 or a.y > self.H + 60
            if not out:
                keep.append(a)
        self.animals = keep

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _make_background(self) -> pygame.Surface:
        bg = pygame.Surface((self.ocean_w, self.H))
        top, bot = C["shallow"], C["deep"]
        for y in range(self.H):
            k = y / self.H
            pygame.draw.line(bg, [int(top[i] + (bot[i] - top[i]) * k) for i in range(3)],
                             (0, y), (self.ocean_w, y))
        rng = random.Random(7)
        for _ in range(140):
            x, y = rng.uniform(0, self.ocean_w), rng.uniform(0, self.H)
            pygame.draw.circle(bg, (40, 120, 155), (int(x), int(y)), rng.choice((1, 1, 2)))
        return bg

    def draw(self):
        scr = self.screen
        scr.blit(self._bg, (0, 0))
        self._draw_waves()

        ov = self._overlay
        ov.fill((0, 0, 0, 0))
        radius = self.cfg.SIM_SAFETY_RADIUS
        for t in self.trash:
            if t in self.blocked:
                pygame.draw.circle(ov, (*C["critical"], 45), (int(t.x), int(t.y)), radius)
                pygame.draw.circle(ov, (*C["critical"], 170), (int(t.x), int(t.y)), radius, 2)
            else:
                pygame.draw.circle(ov, (255, 255, 255, 35), (int(t.x), int(t.y)), radius, 1)
        sc = STATE_COLORS[self.state]
        pygame.draw.circle(ov, (*sc, 28), (int(self.rx), int(self.ry)), radius)
        pygame.draw.circle(ov, (*sc, 150), (int(self.rx), int(self.ry)), radius, 2)
        scr.blit(ov, (0, 0))

        for t in self.trash:
            bob = math.sin(self.t * 1.6 + t.phase) * 6
            img = pygame.transform.rotate(t.sprite, t.angle + bob)
            scr.blit(img, img.get_rect(center=(int(t.x), int(t.y))))
            if t is self.target:
                pygame.draw.circle(scr, C["trash"], (int(t.x), int(t.y)), int(t.size + 8), 2)

        for a in self.animals:
            self._draw_animal(a)

        if self.target is not None and self.state in ("MOVING", "HOLD"):
            self._dashed_line(sc, (self.rx, self.ry), (self.target.x, self.target.y))

        for x, y, t0 in self.effects:
            k = (self.t - t0) / 0.8
            pygame.draw.circle(scr, C["accent"], (int(x), int(y)), int(10 + 40 * k), 2)

        self._draw_robot()
        self._draw_hud()
        self._draw_panel()

    def _draw_waves(self):
        for i in range(9):
            y0 = 40 + i * 80
            pts = [(x, y0 + math.sin(x * 0.02 + self.t * 0.8 + i) * 6)
                   for x in range(0, self.ocean_w + 30, 30)]
            pygame.draw.aalines(self.screen, C["wave"], False, pts)

    def _draw_animal(self, a: Animal):
        scr = self.screen
        if a.kind == "jellyfish":
            pulse = 1 + 0.12 * math.sin(self.t * 3 + a.phase)
            w, h = a.size * 2 * pulse, a.size * 1.3
            x, y = a.x, a.y
            for i in range(5):
                tx = x - w / 2 + 4 + i * (w - 8) / 4
                pts = [(tx + math.sin(self.t * 4 + i + j * 0.7) * 3, y + j * 5) for j in range(6)]
                pygame.draw.lines(scr, (240, 170, 230), False, pts, 2)
            pygame.draw.ellipse(scr, (225, 120, 210), (x - w / 2, y - h, w, h * 1.4))
            pygame.draw.ellipse(scr, (250, 195, 240), (x - w / 3, y - h * 0.8, w / 1.5, h * 0.8))
            pygame.draw.rect(scr, (225, 120, 210), (x - w / 2, y - 1, w, 4))
        else:
            wiggle = math.sin(self.t * 6 + a.phase) * 6
            img = pygame.transform.rotate(a.sprite, -math.degrees(a.heading) + wiggle)
            scr.blit(img, img.get_rect(center=(int(a.x), int(a.y))))

    def _draw_robot(self):
        scr = self.screen
        cos_h, sin_h = math.cos(self.heading), math.sin(self.heading)

        def tf(px, py):
            return (self.rx + px * cos_h - py * sin_h, self.ry + px * sin_h + py * cos_h)

        hull = [tf(*p) for p in ((30, 0), (14, -16), (-24, -16), (-24, 16), (14, 16))]
        pygame.draw.polygon(scr, C["hull"], hull)
        pygame.draw.polygon(scr, (60, 40, 10), hull, 2)
        scoop = [tf(*p) for p in ((30, -12), (40, -16), (40, 16), (30, 12))]
        pygame.draw.polygon(scr, (90, 90, 100), scoop)
        pygame.draw.circle(scr, (40, 60, 80), [int(v) for v in tf(-4, 0)], 9)
        light = STATE_COLORS[self.state]
        if self.state == "HOLD" and int(self.t * 4) % 2:
            light = (255, 255, 255)
        pygame.draw.circle(scr, light, [int(v) for v in tf(-4, 0)], 5)

        if self.state == "HOLD":
            txt = self.f_big.render(f"HOLD {max(0.0, self.hold_timer):0.1f}s", True, C["critical"])
            scr.blit(txt, txt.get_rect(center=(int(self.rx), int(self.ry) - 58)))
            k = 1 - max(0.0, self.hold_timer) / self.cfg.SAFETY_WAIT_SECONDS
            rect = pygame.Rect(0, 0, 76, 76)
            rect.center = (int(self.rx), int(self.ry))
            pygame.draw.arc(scr, C["critical"], rect, math.pi / 2, math.pi / 2 + math.tau * (1 - k), 4)
        elif self.state == "COLLECTING":
            k = 1 - self.collect_timer / self.cfg.SIM_COLLECT_SECONDS
            pygame.draw.circle(scr, C["uncertain"], (int(self.rx), int(self.ry)), int(24 + 14 * k), 2)

    def _draw_hud(self):
        box = pygame.Surface((330, 158), pygame.SRCALPHA)
        box.fill((8, 14, 20, 200))
        self.screen.blit(box, (12, 12))
        x, y = 24, 20
        self._text("OCEAN GUARDIAN  |  SIMULATION", x, y, C["accent"], self.f_med)
        y += 28
        self._text("State:", x, y, C["text"], self.f_med)
        self._text(self.state, x + 80, y, STATE_COLORS[self.state], self.f_med)
        y += 24
        self._text(f"Trash collected   : {self.collected}", x, y, C["text"])
        y += 19
        self._text(f"Animals protected : {self.protected}", x, y, C["marine"])
        y += 19
        m, s = divmod(int(self.t), 60)
        self._text(f"Time              : {m:02d}:{s:02d}", x, y, C["text"])
        y += 21
        if self.state == "HOLD":
            self._text(f"Waiting {max(0.0, self.hold_timer):0.1f}s - {self.hold_reason}"[:44],
                       x, y, C["critical"])
        else:
            self._text(f"Trash: {len(self.trash)}  Animals: {len(self.animals)}  "
                       f"Blocked: {len(self.blocked)}", x, y, C["dim"])

    def _draw_panel(self):
        scr = self.screen
        px = self.ocean_w
        pygame.draw.rect(scr, C["panel"], (px, 0, self.PANEL_W, self.H))
        pygame.draw.line(scr, C["accent"], (px, 0), (px, self.H), 2)
        x, y = px + 16, 14
        self._text("CAMERA", x, y, C["accent"], self.f_med)
        y += 28

        # Camera feed (same boxes/labels as the visualizer) or placeholder.
        feed_w, feed_h = self.PANEL_W - 32, int((self.PANEL_W - 32) * 0.75)
        frame_id, frame = self.bridge.latest_frame()
        if frame is not None and frame_id != self._cam_frame_id:
            h, w = frame.shape[:2]
            surf = pygame.image.frombuffer(frame.tobytes(), (w, h), "RGB")
            self._cam_surface = pygame.transform.smoothscale(surf, (feed_w, feed_h))
            self._cam_frame_id = frame_id
        if self._cam_surface is not None and self.bridge.live:
            scr.blit(self._cam_surface, (x, y))
        else:
            pygame.draw.rect(scr, C["panel_2"], (x, y, feed_w, feed_h))
            self._text("NO CAMERA FEED", x + 100, y + feed_h // 2 - 22, C["dim"], self.f_med)
            self._text("press F = fish  /  T = trash", x + 62, y + feed_h // 2 + 6, C["dim"])
        pygame.draw.rect(scr, C["dim"], (x, y, feed_w, feed_h), 1)
        y += feed_h + 10

        self._text(f"Status : {self.bridge.status}"[:42], x, y, C["text"])
        y += 19
        self._text(f"Input  : {self.cam_source}"[:42], x, y, C["text"])
        y += 22
        r = self.cam_report
        if r is not None:
            self._text(clean_text(r.robot_command)[:38], x, y,
                       COMMAND_COLORS.get(r.command_key, C["text"]), self.f_med)
            y += 24
            self._text(f"Marine {r.marine_count}  Trash {r.trash_count}  "
                       f"Uncertain {r.uncertain_count}", x, y, C["text"])
            y += 19
        else:
            self._text("No camera signal", x, y, C["dim"], self.f_med)
            y += 43
        if self.cam_marine:
            link, col = "Robot link: HOLD (marine life)", C["critical"]
        elif self.cam_trash:
            link, col = "Robot link: COLLECT allowed", C["trash"]
        else:
            link, col = "Robot link: autonomous", C["dim"]
        self._text(link, x, y, col)
        y += 26

        pygame.draw.line(scr, C["panel_2"], (x, y), (px + self.PANEL_W - 16, y), 1)
        y += 10
        self._text("WORLD DECISION ENGINE", x, y, C["accent"])
        y += 20
        wr = self.world_report
        self._text(clean_text(wr.robot_command)[:38], x, y,
                   COMMAND_COLORS.get(wr.command_key, C["text"]), self.f_med)
        y += 24
        self._text(f"Risk {wr.scene_risk}  |  safe {len(wr.collectible)}  "
                   f"blocked {len(wr.blocked)}", x, y, C["text"])
        y += 26

        pygame.draw.line(scr, C["panel_2"], (x, y), (px + self.PANEL_W - 16, y), 1)
        y += 10
        self._text("CONTROLS", x, y, C["accent"])
        y += 20
        for key, desc in (("F", "simulate 'fish detected'"),
                          ("T", "simulate 'trash detected'"),
                          ("C", "clear override"),
                          ("S", "screenshot"),
                          ("Q", "quit (or ESC)")):
            self._text(key, x, y, C["uncertain"], self.f_med)
            self._text(desc, x + 26, y + 2, C["text"])
            y += 20
        self._text("COCO proxies: bird/cat/dog = marine life,", x, self.H - 38, C["dim"])
        self._text("bottle/cup/cell phone = trash", x, self.H - 20, C["dim"])

    # ── Small helpers ─────────────────────────────────────────────────────────

    def _text(self, text: str, x: int, y: int, color, font=None):
        surf = (font or self.f_small).render(text, True, color)
        self.screen.blit(surf, (x, y))

    def _dashed_line(self, color, a, b, dash: int = 10):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length < 1:
            return
        ux, uy = dx / length, dy / length
        for i in range(0, int(length), dash * 2):
            s = (a[0] + ux * i, a[1] + uy * i)
            e = (a[0] + ux * min(i + dash, length), a[1] + uy * min(i + dash, length))
            pygame.draw.line(self.screen, color, s, e, 2)


def run(cfg: Config, use_camera: bool = True, duration: Optional[float] = None):
    Simulation(cfg, use_camera=use_camera, duration=duration).run()
