"""
detector.py — YOLOv8 inference wrapper for Ocean Guardian.

Responsibilities:
  - Load the YOLO model once at startup
  - Run inference on each frame
  - Map raw YOLO class names to Ocean Guardian categories
    (marine / trash / uncertain)
  - Return structured Detection objects

Swapping models:
  - Change Config.MODEL_PATH to your custom .pt file
  - Update Config.MARINE_LIFE_LABELS / TRASH_LABELS to match your classes
  - Everything else stays the same
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

# Ultralytics YOLOv8 — install with: pip install ultralytics
from ultralytics import YOLO

from config import Config


# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Single detected object, fully enriched with domain context."""
    label: str                    # original YOLO class name
    category: str                 # "marine" | "trash" | "uncertain"
    confidence: float             # 0.0 – 1.0
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels
    risk_score: float             # 0.0 = safe, 1.0 = must collect
    action: str                   # recommended robot action string

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


# ── Detector ─────────────────────────────────────────────────────────────────

class OceanGuardianDetector:
    """
    Thin wrapper around YOLOv8 that adds ocean-domain classification.

    Usage:
        detector = OceanGuardianDetector(cfg)
        detections: List[Detection] = detector.detect(frame)
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        print(f"[INFO] Loading model: {cfg.MODEL_PATH}")
        self.model = YOLO(cfg.MODEL_PATH)
        # Build fast lookup sets from config
        self._marine_set = {l.lower() for l in cfg.MARINE_LIFE_LABELS}
        self._trash_set  = {l.lower() for l in cfg.TRASH_LABELS}
        print(f"[INFO] Model loaded. "
              f"Marine labels: {len(self._marine_set)} | "
              f"Trash labels: {len(self._trash_set)}")

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLOv8 inference on a single BGR frame.
        Returns a list of Detection objects sorted by confidence (desc).
        """
        results = self.model(
            frame,
            conf=self.cfg.CONF_THRESHOLD,
            verbose=False,       # suppress per-frame ultralytics logs
        )

        detections: List[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                label = result.names[int(box.cls)]
                conf  = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                category   = self._classify(label, conf)
                risk       = self._risk_score(category, conf)
                action     = self._recommended_action(category, conf)

                detections.append(Detection(
                    label=label,
                    category=category,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    risk_score=risk,
                    action=action,
                ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    # ── Private helpers ───────────────────────────────────────────────────────

    def _classify(self, label: str, conf: float) -> str:
        """
        Map a YOLO label to an ocean domain category.
        Confidence-gated: low-confidence detections become 'uncertain'
        regardless of the label.
        """
        if conf < self.cfg.UNCERTAIN_BELOW:
            return "uncertain"

        label_lower = label.lower()
        if label_lower in self._marine_set:
            return "marine"
        if label_lower in self._trash_set:
            return "trash"
        return "uncertain"

    def _risk_score(self, category: str, conf: float) -> float:
        """
        Risk score represents how urgently this object should be collected.
        0.0 = definitely avoid (living creature)
        0.5 = unclear
        1.0 = definitely collect (confirmed trash)
        """
        if category == "marine":
            return round(max(0.0, 0.1 - conf * 0.1), 3)  # near zero
        if category == "trash":
            return round(min(1.0, conf * 1.1), 3)          # scales with confidence
        return 0.5  # uncertain

    def _recommended_action(self, category: str, conf: float) -> str:
        if category == "marine":
            return "⛔  AVOID — Marine Life"
        if category == "trash":
            if conf >= self.cfg.HIGH_CONF:
                return "♻️  COLLECT — Confirmed Trash"
            return "♻️  COLLECT — Likely Trash"
        return "⚠️  HOLD — Uncertain Object"
