"""
config.py — Central configuration for Ocean Guardian.

All tunable parameters live here so the rest of the code stays clean.
Swap models, adjust thresholds, or reclassify labels in one place.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ── Color palette (BGR for OpenCV) ──────────────────────────────────────────
# Bioluminescent ocean-inspired palette

COLORS = {
    "marine":    (0, 210, 140),   # teal-green  → safe / life
    "trash":     (0, 80,  220),   # vivid red   → collect
    "uncertain": (0, 165, 255),   # amber       → caution
    "critical":  (0, 0,   255),   # pure red    → do not collect (mixed scene)
    "hud_bg":    (15, 20, 25),    # near-black  → HUD background
    "hud_text":  (220, 240, 230), # off-white   → HUD text
    "accent":    (0, 220, 160),   # bright teal → highlights
    "panel_bg":  (20, 28, 35),    # dark navy   → side panel
}


@dataclass
class Config:
    # ── Model ────────────────────────────────────────────────────────────────
    # Default: YOLOv8n (nano, fast) trained on COCO.
    # For hackathon demo, COCO labels are remapped to marine / trash categories.
    # Replace with your custom-trained model path for real deployment:
    #   MODEL_PATH = "models/ocean_guardian_v1.pt"
    MODEL_PATH: str = "yolov8n.pt"

    # ── Detection thresholds ─────────────────────────────────────────────────
    CONF_THRESHOLD: float = 0.40   # minimum confidence to show a detection
    UNCERTAIN_BELOW: float = 0.55  # below this → "uncertain" even if classified
    HIGH_CONF: float = 0.70        # above this → high-confidence decision

    # ── Class label mapping ───────────────────────────────────────────────────
    # Maps COCO class names to ocean domain categories.
    # When you train a custom model, replace with your real label set.
    #
    # CUSTOM TRAINING LABELS (recommended):
    #   Marine life:  fish, jellyfish, coral, crab, starfish, sea_turtle,
    #                 seahorse, octopus, dolphin, whale, shark, ray
    #   Trash:        plastic_bag, plastic_bottle, fishing_net, can,
    #                 styrofoam, rope, glass_bottle, tire, metal_debris
    #   Uncertain:    rock, seaweed, driftwood, shell (natural but non-living)

    MARINE_LIFE_LABELS: List[str] = field(default_factory=lambda: [
        # COCO animals used as proxies for marine life in webcam demo
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
        "bear", "zebra", "giraffe",
        # Real target labels when custom model is trained:
        "fish", "jellyfish", "coral", "crab", "starfish", "sea_turtle",
        "seahorse", "octopus", "dolphin", "whale", "shark", "ray",
    ])

    TRASH_LABELS: List[str] = field(default_factory=lambda: [
        # COCO objects used as trash proxies for demo
        "bottle", "cup", "fork", "knife", "spoon", "bowl",
        "chair", "couch", "potted plant", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "book",
        "scissors", "toothbrush", "hair drier", "vase",
        # Real target labels when custom model is trained:
        "plastic_bag", "plastic_bottle", "fishing_net", "can",
        "styrofoam", "rope", "glass_bottle", "tire", "metal_debris",
    ])

    UNCERTAIN_LABELS: List[str] = field(default_factory=lambda: [
        # Ambiguous objects
        "rock", "seaweed", "driftwood", "shell", "person",
        "backpack", "umbrella", "handbag", "tie", "suitcase",
    ])

    # ── Decision engine parameters ────────────────────────────────────────────
    # Proximity threshold: if a trash box overlaps or is within this many pixels
    # of a marine life box, flag as DO NOT COLLECT for safety.
    PROXIMITY_SAFETY_MARGIN_PX: int = 60

    # How many trash items trigger "priority cleanup zone"
    PRIORITY_ZONE_THRESHOLD: int = 3

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_EVERY_N_FRAMES: int = 30   # print status to terminal every N frames

    # ── Visual layout ─────────────────────────────────────────────────────────
    PANEL_WIDTH: int = 320          # width of the right-side HUD panel
    FONT_SCALE_LABEL: float = 0.55
    FONT_SCALE_HUD: float = 0.52
    BOX_THICKNESS: int = 2

    def banner(self) -> str:
        return f"""
╔══════════════════════════════════════════════════════════════╗
║              OCEAN GUARDIAN  —  v1.0 Prototype               ║
║         Ecosystem-Safe AI Ocean Cleanup Assistant            ║
╠══════════════════════════════════════════════════════════════╣
║  Model   : {self.MODEL_PATH:<48}║
║  Min Conf: {self.CONF_THRESHOLD:<48.2f}║
╚══════════════════════════════════════════════════════════════╝"""
