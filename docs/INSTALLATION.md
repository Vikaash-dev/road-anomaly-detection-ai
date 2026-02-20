# Installation Guide

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | ≥ 3.9 |
| OS | 64-bit Raspberry Pi OS (Bullseye/Bookworm) or any Linux/macOS/Windows |
| RAM | ≥ 2 GB (4 GB recommended on Raspberry Pi 4) |
| Storage | ≥ 500 MB free |

---

## 1. Clone the Repository

```bash
git clone https://github.com/Vikaash-dev/road-anomaly-detection-ai.git
cd road-anomaly-detection-ai
```

---

## 2. Create a Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate       # Linux / macOS / Raspberry Pi OS
# venv\Scripts\activate.bat   # Windows
```

---

## 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### What gets installed

| Package | Purpose |
|---------|---------|
| `opencv-python` | Video capture, frame processing, annotation |
| `numpy` | Tensor manipulation and NMS |
| `pandas` | Optional CSV analysis |
| `onnxruntime` | CPU inference for the ONNX pothole model |
| `ultralytics` | YOLO11 PyTorch obstacle model (`main.py` / `optimized_main.py`) |

> **Note:** `ultralytics` is only required for `main.py` and `optimized_main.py`.
> `ultimate_pipeline.py` works with ONNX Runtime only.

---

## 4. Verify the Installation

```bash
python3 -c "import cv2, numpy, onnxruntime, ultralytics; print('All dependencies OK')"
```

Expected output:

```
All dependencies OK
```

---

## 5. Raspberry Pi 4 — Specific Notes

### Install system-level dependencies first

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv libatlas-base-dev libopenblas-dev
```

### Use a lightweight OpenCV build (optional, saves RAM)

```bash
pip install opencv-python-headless>=4.8.0
```

### Verify ONNX Runtime ARM support

```bash
python3 -c "import onnxruntime as ort; print(ort.get_device())"
# Expected: CPU
```

---

## 6. Model Files

The following model files must be present in the repository root:

| File | Description |
|------|-------------|
| `best.onnx` | Custom-trained YOLO11n for pothole detection |
| `yolo11n.pt` | COCO-pretrained YOLO11n for obstacle detection |

Both files are included in this repository.

---

## 7. Uninstall

```bash
pip uninstall opencv-python numpy pandas onnxruntime ultralytics -y
```
