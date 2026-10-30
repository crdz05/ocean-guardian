"""
╔══════════════════════════════════════════════════════════════╗
║           OCEAN GUARDIAN — Ecosystem-Safe AI Cleanup         ║
║      Real-time marine life vs. trash detection system        ║
╚══════════════════════════════════════════════════════════════╝

Entry point. Run this file to start the demo:
    python main.py                     → original camera mode (YOLO + OpenCV)
    python main.py --sim               → pygame simulation + webcam in parallel
    python main.py --sim --no-camera   → simulation only (F/T keys as "camera")

Optional flags:
    --source 0          → webcam index (default: 0)
    --source video.mp4  → video file instead of webcam (camera mode only)
    --model custom.pt   → swap in a custom-trained YOLO model
    --conf 0.4          → minimum confidence threshold (default: 0.4)
    --no-gui            → print-only mode (no OpenCV window, camera mode only)
    --duration 30       → auto-quit the simulation after N seconds
"""

import argparse
import sys

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
    parser.add_argument("--sim", action="store_true",
                        help="Run the pygame simulation (with the webcam in parallel)")
    parser.add_argument("--no-camera", action="store_true",
                        help="With --sim: skip the webcam/YOLO thread (use F/T keys)")
    parser.add_argument("--duration", type=float, default=None,
                        help="With --sim: quit automatically after N seconds")
    return parser.parse_args()


def open_source(source_arg: str):
    """Open webcam or video file. Returns cv2.VideoCapture."""
    import cv2

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


def run_simulation(args, cfg: Config):
    try:
        import simulation
    except ImportError as exc:
        print(f"[ERROR] Could not start the simulation: {exc}")
        print("        Install pygame with:  pip install pygame-ce")
        sys.exit(1)

    if args.source.isdigit():
        cfg.CAMERA_INDEX = int(args.source)
    simulation.run(cfg, use_camera=not args.no_camera, duration=args.duration)


def run_camera(args, cfg: Config):
    import cv2

    try:
        from detector import OceanGuardianDetector
        detector = OceanGuardianDetector(cfg)       # YOLO wrapper
    except Exception as exc:   # ImportError, OSError (e.g. WinError 4551), ...
        print(f"\n[ERROR] YOLO detector unavailable: {type(exc).__name__}: {exc}")
        print("        torch / ultralytics could not be loaded on this machine.")
        print("        Run the simulation instead:  python main.py --sim --no-camera")
        sys.exit(1)

    from visualizer import Visualizer
    from decision_engine import DecisionEngine

    # ── Core components ────────────────────────────────────────
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


def main():
    # Box-drawing characters / emoji in logs: avoid UnicodeEncodeError on
    # Windows consoles or when output is piped.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    args = parse_args()
    cfg = Config()

    # Allow CLI overrides of config defaults
    if args.model:
        cfg.MODEL_PATH = args.model
    if args.conf is not None:
        cfg.CONF_THRESHOLD = args.conf

    print(cfg.banner())

    if args.sim:
        run_simulation(args, cfg)
    else:
        run_camera(args, cfg)


if __name__ == "__main__":
    main()
