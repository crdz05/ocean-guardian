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

## Technologies

Python, YOLOv8 (Ultralytics), OpenCV, NumPy

## My role

Co-developed the prototype with a teammate. I directed the implementation of the YOLOv8 detection pipeline, designed the proximity-based decision engine and its priority logic, and directed an AI-assisted development process that structured the system into the layers above.
