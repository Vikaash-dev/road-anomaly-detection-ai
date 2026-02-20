# API Reference

---

## main.py / optimized_main.py

Both scripts share the same `DualDetector` class. `optimized_main.py` additionally
provides the `VideoStream` class for threaded frame capture.

---

### class `DualDetector`

```python
DualDetector(
    pothole_model_path,
    obstacle_model_path=None,
    conf_threshold=0.75,
    iou_threshold=0.4,
    input_size=640,
    camera_mode=False
)
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pothole_model_path` | `str` | — | Path to the ONNX pothole detection model |
| `obstacle_model_path` | `str` | `None` | Path to the YOLO11 PT obstacle model (optional) |
| `conf_threshold` | `float` | `0.75` | Minimum confidence score to accept a detection |
| `iou_threshold` | `float` | `0.4` | IOU threshold used in Non-Maximum Suppression |
| `input_size` | `int` | `640` | Model input resolution (auto-detected from ONNX graph) |
| `camera_mode` | `bool` | `False` | When `True`, reads from webcam instead of a video file |

---

#### `preprocess_pothole(image)`

Prepares a BGR frame for ONNX inference.

**Parameters:** `image` — `np.ndarray` (H × W × 3, BGR)

**Returns:** `(input_tensor, scale_x, scale_y)`

| Return Value | Type | Description |
|--------------|------|-------------|
| `input_tensor` | `np.ndarray` (1×3×H×W, float32) | Normalised, channel-first batch tensor |
| `scale_x` | `float` | Horizontal scale factor for coordinate back-projection |
| `scale_y` | `float` | Vertical scale factor for coordinate back-projection |

---

#### `postprocess_pothole(outputs, scale_x, scale_y, original_shape)`

Parses YOLO raw output and applies NMS.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `outputs` | `list[np.ndarray]` | Raw ONNX session output |
| `scale_x` | `float` | From `preprocess_pothole` |
| `scale_y` | `float` | From `preprocess_pothole` |
| `original_shape` | `tuple` | `(height, width)` of the original frame |

**Returns:** `list[list]` — Each element is `[x1, y1, x2, y2, confidence, class_id]`

---

#### `detect_obstacles(frame)`

Runs the YOLO11 PT model to detect obstacles (vehicles, persons, animals).

**Parameters:** `frame` — `np.ndarray` (H × W × 3, BGR)

**Returns:** `list[list]` — Each element is `[x1, y1, x2, y2, confidence, class_id]`

> Returns `[]` if `obstacle_model` is `None` or inference fails.

---

#### `calculate_pothole_diameter(bbox)`

Estimates the approximate diameter of a pothole.

**Parameters:** `bbox` — `list[int]` — `[x1, y1, x2, y2]`

**Returns:** `float` — `(width + height) / 2` in pixels

---

#### `calculate_iou(box1, box2)`

Computes the Intersection over Union between two axis-aligned bounding boxes.

**Parameters:** `box1`, `box2` — `list[float]` — `[x_min, y_min, x_max, y_max]`

**Returns:** `float` in range `[0.0, 1.0]`

---

#### `track_and_calculate_motion(current_detections)`

Updates IoU-based vehicle tracks and returns a motion status for each detection.

**Parameters:** `current_detections` — `list[list]` — obstacle detections for the current frame

**Returns:** `dict[int, str]` — Maps detection index to one of:

| Status | Meaning |
|--------|---------|
| `"Moving"` | Average centroid displacement > 15 px over last 3 frames |
| `"Stationary"` | Average centroid displacement ≤ 15 px over last 3 frames |
| `"Unknown"` | Track seen for fewer than 3 frames |
| `"N/A"` | Detection is not a vehicle class |

---

#### `log_detections_to_csv(frame_number, pothole_detections, obstacle_detections, motion_statuses)`

Appends one row to the internal CSV buffer.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `frame_number` | `int` | Current frame index |
| `pothole_detections` | `list[list]` | From `postprocess_pothole` |
| `obstacle_detections` | `list[list]` | From `detect_obstacles` |
| `motion_statuses` | `dict[int, str]` | From `track_and_calculate_motion` |

---

#### `save_csv(csv_path)`

Writes the in-memory CSV buffer to disk.

**Parameters:** `csv_path` — `str` — destination file path

**CSV Schema**

| Column | Type | Description |
|--------|------|-------------|
| `Serial_Number` | int | Auto-incrementing row ID |
| `Frame_Number` | int | Video frame index |
| `Total_Potholes` | int | Potholes detected in this frame |
| `Total_Obstacles` | int | Obstacles detected in this frame |
| `Pothole_Details` | str | Pipe-separated: `Class:conf=X,bbox=(…),diameter=Ypx` |
| `Obstacle_Details` | str | Pipe-separated: `Class:conf=X,bbox=(…),motion=Status` |

---

#### `draw_detections(image, pothole_detections, obstacle_detections, fps=0)`

Annotates a frame with bounding boxes and a statistics panel.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | `np.ndarray` | BGR frame |
| `pothole_detections` | `list[list]` | Drawn in **red** with diameter label |
| `obstacle_detections` | `list[list]` | Drawn in **green** with class + confidence |
| `fps` | `float` | Displayed in the stats panel |

**Returns:** `np.ndarray` — Annotated frame with 200 px stats panel prepended at the top

---

#### `process_video(video_source, output_path=None, display=True)`

Main processing loop.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `video_source` | `str` | Path to input video file (ignored in `camera_mode`) |
| `output_path` | `str` | Path for output MP4; also derives CSV path if not set separately |
| `display` | `bool` | Show live preview window |

---

### class `VideoStream` _(optimized_main.py only)_

Threaded video capture wrapper that eliminates blocking reads on the inference thread.

```python
VideoStream(src=0)
```

| Method | Description |
|--------|-------------|
| `start()` | Launches background reader thread; returns `self` |
| `read()` | Returns `(grabbed, frame)` — most recently captured frame |
| `stop()` | Signals the reader thread to exit and releases the capture |
| `isOpened()` | Delegates to `cv2.VideoCapture.isOpened()` |
| `get(propId)` | Delegates to `cv2.VideoCapture.get()` |

---

## ultimate_pipeline.py

### class `UltimateDetector`

Lightweight single-model pipeline. No `ultralytics` dependency.

```python
UltimateDetector(
    pothole_model_path,
    conf_threshold=0.3,
    iou_threshold=0.4,
    input_size=416,
    camera_mode=False
)
```

#### `letterbox(img, new_shape=(640, 640), color=(114, 114, 114))`

Resizes with aspect-ratio preservation and pads to `new_shape`.

Also applies pre-sharpening (unsharp mask) and `INTER_AREA` downscaling.

**Returns:** `(padded_img, ratio, (dw, dh))`

#### `preprocess_pothole(image)`

Calls `letterbox`, converts to RGB, normalises to Float32, transposes to CHW, adds batch dim.

**Returns:** `(input_tensor, ratio, dw, dh)`

#### `postprocess_pothole(outputs, ratio, dw, dh, original_shape)`

Parses YOLO output tensors, rescales boxes from letterbox space back to original resolution, applies NMS.

**Returns:** `list[list]` — `[x1, y1, x2, y2, confidence, class_id]`

#### `process_video(video_source, output_path=None)`

Main loop with temporal blending and adaptive CLAHE pre-processing before inference.
