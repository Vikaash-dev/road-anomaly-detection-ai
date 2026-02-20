# Configuration Reference

All runtime parameters are passed as command-line arguments. There is no separate
configuration file; defaults are defined in the `argparse` section of each script.

---

## Common Parameters

These arguments are available in all three scripts.

### `--pothole-model` _(required)_

Path to the ONNX pothole detection model.

```bash
--pothole-model best.onnx
```

The model input size is read automatically from the ONNX graph. If the shape is dynamic,
the default `input_size` (640 for `main.py` / `optimized_main.py`, 416 for
`ultimate_pipeline.py`) is used.

---

### `--video` _(required)_

Path to the input MP4 dashcam video.

```bash
--video demo1.mp4
```

Accepts any video format supported by the installed OpenCV build (MP4, AVI, MKV, etc.).

---

### `--output` _(optional)_

Path for the annotated output video. Omit to run inference without saving video.

```bash
--output output_demo1.mp4
```

The output video is written with the `mp4v` codec at the **target FPS** (5 FPS), not the
source FPS. Output resolution = source width × (source height + 200 px stats panel).

---

## main.py / optimized_main.py — Additional Parameters

### `--obstacle-model` _(optional)_

Path to the YOLO11 PT obstacle model.

```bash
--obstacle-model yolo11n.pt
```

When omitted, obstacle detection is disabled and only potholes are reported.

---

### `--csv` _(optional)_

Path for the detection CSV log file. When omitted, a CSV is still auto-generated in the
same directory as `--output` (filename: `<output_stem>_detections.csv`).

```bash
--csv detections.csv
```

---

### `--conf` _(optional, default: `0.75`)_

Confidence threshold applied to **pothole** detections. Range: `0.0` – `1.0`.

```bash
--conf 0.75
```

The obstacle model uses a slightly lower effective threshold:
`max(0.25, conf × 0.7)` to improve recall of distant vehicles.

| Value | Effect |
|-------|--------|
| `0.90` | Very few detections; only high-certainty potholes |
| `0.75` | Default; good balance of precision and recall |
| `0.50` | More detections; possible false positives |
| `0.30` | Aggressive; use only for testing |

---

### `--iou` _(optional, default: `0.4`)_

IOU threshold for Non-Maximum Suppression. Range: `0.0` – `1.0`.

```bash
--iou 0.4
```

| Value | Effect |
|-------|--------|
| `0.6` | Allows more overlapping boxes (use for dense potholes) |
| `0.4` | Default; suppresses most duplicate detections |
| `0.2` | Aggressive suppression; risk of merging nearby potholes |

---

### `--camera` _(optional flag)_

When present, reads from the default webcam (`/dev/video0`) instead of a file.

```bash
--camera
```

Camera resolution is fixed at 640×480. The `--video` argument is ignored when `--camera` is set.

---

## Tuning Tips

### Maximise recall (catch more anomalies)

```bash
python main.py --pothole-model best.onnx --obstacle-model yolo11n.pt \
  --video input.mp4 --conf 0.5 --iou 0.5
```

### Maximise precision (reduce false positives)

```bash
python main.py --pothole-model best.onnx --obstacle-model yolo11n.pt \
  --video input.mp4 --conf 0.85 --iou 0.35
```

### Pothole-only mode (fastest)

```bash
python ultimate_pipeline.py --pothole-model best.onnx --video input.mp4 --output out.mp4
```

---

## Internal Constants

The following constants are set in the source code and require a code edit to change.

| Constant | File | Default | Description |
|----------|------|---------|-------------|
| `target_fps` | `main.py`, `optimized_main.py` | `5.0` | Output frames per second |
| `motion_threshold` | `main.py`, `optimized_main.py` | `15 px` | Centroid displacement to classify as Moving |
| `motion_history_length` | `main.py`, `optimized_main.py` | `5` | Frames kept in vehicle track history |
| `iou_match_threshold` | `main.py`, `optimized_main.py` | `0.3` | Min IoU to associate a detection with an existing track |
| `track_prune_age` | `main.py`, `optimized_main.py` | `10` | Frames before an unmatched track is removed |
| `clahe_clip_limit` | `ultimate_pipeline.py` | `2.0` | CLAHE contrast clip limit |
| `temporal_blend_alpha` | `ultimate_pipeline.py` | `0.8` | Current frame weight in temporal blend |
