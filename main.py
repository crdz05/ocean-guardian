"""
╔══════════════════════════════════════════════════════════════╗
║           OCEAN GUARDIAN — Ecosystem-Safe AI Cleanup         ║
║      Real-time marine life vs. trash detection system        ║
╚══════════════════════════════════════════════════════════════╝

Entry point. Run this file to start the demo:
    python main.py

Optional flags:
    --source 0          → webcam index (default: 0)
    --source video.mp4  → video file instead of webcam
    --model custom.pt   → swap in a custom-trained YOLO model
    --conf 0.4          → minimum confidence threshold (default: 0.4)
    --no-gui            → print-only mode (no OpenCV window)
"""

import argparse
import sys
import cv2

from detector import OceanGuardianDetector
from visualizer import Visualizer
from decision_engine import DecisionEngine
from config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Ocean Guardian — Real-time Marine AI")
    parser.add_argument("--source", default="0",
                        help="Camera index (0) or path to video file")
    parser.add_argument("--model", default=None,
                        help="Path to custom .pt model file (overrides config default)")
    parser.add_argument("--conf", type=float, default=None,
                        help="Detection confidence threshold (0.0–1.0)")
    parser.add_argument("--no-gui", action="store_true",
                        help="Disable OpenCV window (terminal output only)")
    return parser.parse_args()


def open_source(source_arg: str):
    """Open webcam or video file. Returns cv2.VideoCapture."""
    # Try to parse as integer (webcam index) first
    try:
        src = int(source_arg)
    except ValueError:
        src = source_arg  # treat as file path

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source_arg}")
        sys.exit(1)

    # Webcam resolution hint — higher res = better detection
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def main():
    args = parse_args()
    cfg = Config()

    # Allow CLI overrides of config defaults
    if args.model:
        cfg.MODEL_PATH = args.model
    if args.conf is not None:
        cfg.CONF_THRESHOLD = args.conf

    print(cfg.banner())

    # ── Core components ────────────────────────────────────────
    detector = OceanGuardianDetector(cfg)       # YOLO wrapper
    decision = DecisionEngine(cfg)              # robot decision logic
    viz = Visualizer(cfg)                       # OpenCV overlay renderer

    cap = open_source(args.source)

    print("[INFO] Starting detection loop. Press Q to quit.\n")

    frame_count = 0
    show_gui = not args.no_gui

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream or failed frame read.")
            break

        frame_count += 1

        # ── 1. Run detection ───────────────────────────────────
        detections = detector.detect(frame)

        # ── 2. Decision engine evaluates the scene ─────────────
        scene_report = decision.evaluate(detections)

        # ── 3. Render overlay on the frame ─────────────────────
        output_frame = viz.render(frame, detections, scene_report)

        # ── 4. Terminal status every N frames ──────────────────
        if frame_count % cfg.LOG_EVERY_N_FRAMES == 0:
            decision.print_status(scene_report, frame_count)

        # ── 5. Display ─────────────────────────────────────────
        if show_gui:
            cv2.imshow("Ocean Guardian", output_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:   # Q or ESC to quit
                break
            elif key == ord("s"):              # S to save screenshot
                fname = f"capture_frame_{frame_count}.jpg"
                cv2.imwrite(fname, output_frame)
                print(f"[INFO] Screenshot saved → {fname}")

    cap.release()
    if show_gui:
        cv2.destroyAllWindows()

    print("\n[INFO] Session ended.")
    decision.print_summary()


if __name__ == "__main__":
    main()
