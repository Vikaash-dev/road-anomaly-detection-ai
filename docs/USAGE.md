# Usage Guide

This project provides three scripts, each suited to a different deployment scenario.

| Script | Models Used | Best For |
|--------|-------------|----------|
| `main.py` | ONNX + PyTorch | Full dual-model detection |
| `optimized_main.py` | ONNX + PyTorch | Same as above, faster frame capture |
| `ultimate_pipeline.py` | ONNX only | Minimal dependencies, max speed |

---

## main.py — Dual-Model Detector

Detects potholes (ONNX) and obstacles such as vehicles, pedestrians, and animals (YOLO11 PT).

### Basic usage

```bash
python main.py \
  --pothole-model best.onnx \
  --obstacle-model yolo11n.pt \
  --video demo1.mp4 \
  --output output.mp4 \
  --csv detections.csv
```

### Command-line arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--pothole-model` | ✅ | — | Path to the ONNX pothole model |
| `--obstacle-model` | ❌ | None | Path to the YOLO11 PT obstacle model |
| `--video` | ✅ | — | Path to the input MP4 video file |
| `--output` | ❌ | None | Path to save annotated output video |
| `--csv` | ❌ | None | Path to save the detection CSV log |
| `--conf` | ❌ | 0.75 | Confidence threshold (0.0 – 1.0) |
| `--iou` | ❌ | 0.4 | IOU threshold for NMS |
| `--camera` | ❌ | False | Use webcam instead of video file |

### Example: pothole detection only (no obstacle model)

```bash
python main.py \
  --pothole-model best.onnx \
  --video demo2.mp4 \
  --output output_demo2.mp4
```

### Example: live camera feed

```bash
python main.py \
  --pothole-model best.onnx \
  --obstacle-model yolo11n.pt \
  --camera
```

---

## optimized_main.py — Multithreaded Version

Identical interface to `main.py` but uses a background thread to read frames, which reduces I/O stalls on Raspberry Pi 4.

```bash
python optimized_main.py \
  --pothole-model best.onnx \
  --obstacle-model yolo11n.pt \
  --video demo1.mp4 \
  --output output_optimized.mp4 \
  --csv detections_optimized.csv
```

Use this script when you need the most consistent frame rate while still using both models.

---

## ultimate_pipeline.py — Ultra-Optimized Pipeline

Single ONNX model with advanced image enhancement. No `ultralytics` dependency required.

```bash
python ultimate_pipeline.py \
  --pothole-model best.onnx \
  --video demo1.mp4 \
  --output output_ultimate.mp4
```

### Command-line arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--pothole-model` | ✅ | — | Path to the ONNX model |
| `--video` | ✅ | — | Path to the input MP4 video file |
| `--output` | ❌ | None | Path to save annotated output video |

---

## Output Files

### Annotated video

The output MP4 includes:

- **Red bounding boxes** — potholes with confidence score and diameter label
- **Green bounding boxes** — obstacles (vehicles, persons, animals) with class and confidence
- **Statistics panel** (top of frame) — live FPS, frame count, total detection counts

### Detection CSV

Each processed frame produces one row:

```
Serial_Number,Frame_Number,Total_Potholes,Total_Obstacles,Pothole_Details,Obstacle_Details
1,1,2,1,"Pothole:conf=0.89,bbox=(479,591,726,688),diameter=172.0px | ...",car:conf=0.81,...
```

See [API.md](API.md) for the full column schema.

---

## Running on the Demo Videos

```bash
# Demo 1 — Rural road with potholes
python main.py --pothole-model best.onnx --obstacle-model yolo11n.pt \
  --video demo1.mp4 --output output_demo1.mp4 --csv demo1_results.csv

# Demo 2 — Highway with vehicles and potholes
python main.py --pothole-model best.onnx --obstacle-model yolo11n.pt \
  --video demo2.mp4 --output output_demo2.mp4 --csv demo2_results.csv

# Demo 3 — Rural road with animals and vehicles
python main.py --pothole-model best.onnx --obstacle-model yolo11n.pt \
  --video demo3.mp4 --output output_demo3.mp4 --csv demo3_results.csv
```

---

## Keyboard Shortcuts (during video display)

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `s` | Save a screenshot of the current frame |

---

## Tips

- For best performance on Raspberry Pi 4, use `optimized_main.py` or `ultimate_pipeline.py`.
- Lower `--conf` (e.g. `0.5`) if you want more detections; raise it (e.g. `0.85`) to reduce false positives.
- Set `--output` to a path on a fast storage device (e.g. USB 3.0 SSD) to avoid write bottlenecks.
