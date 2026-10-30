# Ocean Guardian

Ecosystem-aware computer-vision prototype for an ocean-cleanup robot. Built at HackUSF (University of South Florida, March 2026) by a team of two.

## Problem

An ocean-cleanup robot should collect trash without harming the marine life around it. A robot that only detects trash cannot tell when collecting is unsafe.

## Solution

Ocean Guardian runs a real-time loop over a webcam or video file. YOLOv8 detects objects in each frame, a decision engine reasons about the whole scene, and an OpenCV overlay shows the result. The key rule: any trash that is close to marine life is blocked from collection.

No robot hardware is connected. The decision engine produces a command string that is displayed on screen and logged to the terminal.

## How it works

1. **Detection** (`detector.py`): wraps YOLOv8 and maps each class label to a category: marine life, trash, or uncertain. Detections below a confidence of 0.55 are treated as uncertain.
2. **Decision logic** (`decision_engine.py`): flags trash as blocked when its box overlaps a marine-life box, or when the centers are closer than 60 pixels (configurable). It then picks one scene-level outcome, in priority order:
   1. HOLD: trash is near marine life
   2. OBSERVE: marine life only
   3. PRIORITY CLEANUP: 3 or more collectible trash items
   4. COLLECT: collectible trash and no marine life
   5. PROCEED WITH CARE: collectible trash and marine life that is not close
   6. SCANNING: only uncertain objects
   7. CLEAR: nothing detected
3. **Visualization** (`visualizer.py`): draws boxes, labels, and a side panel with counts, scene risk, and the current command.

`config.py` holds every tunable parameter, and `main.py` runs the loop. Because the decision logic works on categories instead of raw YOLO class names, the model can be swapped by changing the weights path and the label lists.

## Current limitations

- The demo uses the pretrained `yolov8n.pt` model (COCO). COCO animals such as birds, cats, and dogs stand in for marine life, and everyday objects such as bottles and cups stand in for trash. The model is not trained on ocean imagery.
- Training a custom model is the next step. `data.yaml` defines a 15-class target label set (marine life, trash, and uncertain objects), but no dataset or trained weights are included.
- No accuracy evaluation is included.
- Distances are measured in image pixels, not real-world units.

## Run it

```bash
pip install -r requirements.txt

# Webcam
python main.py

# Video file
python main.py --source video.mp4

# Custom model and confidence threshold
python main.py --model path/to/model.pt --conf 0.5

# Terminal output only
python main.py --no-gui
```

Press `Q` or `ESC` to quit and `S` to save a screenshot. Ultralytics downloads `yolov8n.pt` on first run.

## Simulation demo

`simulation.py` is a top-down pygame view of a collector robot. It runs together with the webcam detector:

- The robot drives to the nearest trash (bottles, bags, nets) and collects it, which increments **Trash collected**.
- Animals (fish, turtles, jellyfish) appear at random and drift slowly.
- **Safety rule:** if an animal is inside the safety radius (`SIM_SAFETY_RADIUS`, 90 px) of the robot or of its target trash, the robot stops in **HOLD**. It shows a countdown of `SAFETY_WAIT_SECONDS` (3 s), then rechecks. If the animal has left, it collects. If not, it waits again or switches to another safe trash item. Each wait adds to **Animals protected**.
- The simulated world is evaluated by the same `DecisionEngine` used in camera mode. The sim objects become `Detection`s, so the same proximity rule that blocks trash in camera mode blocks it here.
- The HUD shows the state (SCANNING / MOVING / HOLD / COLLECTING), trash collected, animals protected, and elapsed time.

**Camera in parallel.** A background thread opens the webcam, runs `detector.py` and the `DecisionEngine`, and shows the feed with boxes and labels in the right panel:

- Marine life on camera puts the robot in **HOLD**.
- Trash on camera (COLLECT / PRIORITY CLEANUP) allows collection and drops a new trash item next to the robot.

### Run it

```bash
pip install -r requirements.txt

python main.py --sim               # simulation + webcam detector
python main.py --sim --no-camera   # simulation only (keys act as the camera)
python main.py                     # original camera mode
python main.py --sim --duration 30 # auto-quit after 30 s
```

### Controls

| Key | Action |
|-----|--------|
| `F` | Simulate "fish detected" on camera (robot goes to HOLD) |
| `T` | Simulate "trash detected" on camera (COLLECT) |
| `C` | Clear the F/T override |
| `S` | Save a screenshot (`sim_screenshot_*.png`) |
| `Q` / `ESC` | Quit |

The F/T overrides take priority over the live camera. They feed synthetic detections through the real `DecisionEngine`, so the panel shows the engine's own command.

### COCO proxies (important for the live camera)

`yolov8n.pt` is trained on COCO, which has **no "fish" class**. A photo of a fish shown to the webcam is usually not detected as marine life, or is detected as `bird`. `config.py` uses COCO classes as stand-ins:

- **Marine life:** `bird`, `cat`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe`
- **Trash:** `bottle`, `cup`, `cell phone`, `book`, `laptop`, `mouse`, `remote`, `scissors`, ...

**Display alias:** in practice a fish shown to the webcam is detected as `bird`. `Config.PROXY_DISPLAY_NAMES` therefore shows every COCO animal proxy as **`fish`** on screen. The original class stays in `Detection.raw_label`, and the simulation log prints it, e.g. `HOLD 3s - Camera: fish 72% (COCO: bird)`. Detections below 0.55 confidence stay `uncertain` and keep their raw label. Set `PROXY_DISPLAY_NAMES = {}` to show the raw COCO labels.

For a live demo, show the camera a picture of a fish (usually seen as `bird`), bird, cat or dog to trigger HOLD, and a bottle, cup or phone to trigger COLLECT. Use the `F` key for a reliable "fish detected". A custom model trained on the `data.yaml` labels (including `fish`) would remove the need for proxies.

### If torch is blocked (Windows "WinError 4551")

Some Windows machines block torch DLLs through an Application Control policy. `ultralytics` is imported lazily inside `OceanGuardianDetector`, so the simulation still starts: the camera thread prints a warning and the robot runs with `F`/`T` as the camera input. The original camera mode (`python main.py`) needs the detector and exits with a hint to use `--sim --no-camera`.

## Technologies

Python, YOLOv8 (Ultralytics), OpenCV, NumPy, pygame

## My role

Co-developed the prototype with a teammate. I directed the implementation of the YOLOv8 detection pipeline, designed the proximity-based decision engine and its priority logic, and directed an AI-assisted development process that structured the system into the layers above.
