#!/usr/bin/env python3
import cv2
import numpy as np
import onnxruntime as ort
import time
import os
import argparse
import threading

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.grabbed, self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()
        
    def isOpened(self):
        return self.stream.isOpened()
        
    def get(self, propId):
        return self.stream.get(propId)

class UltimateDetector:
    def __init__(self, pothole_model_path, conf_threshold=0.3, iou_threshold=0.4, input_size=416, camera_mode=False):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.camera_mode = camera_mode

        print("[INFO] Loading Pothole ONNX model...")
        self.pothole_session = ort.InferenceSession(pothole_model_path, providers=['CPUExecutionProvider'])
        self.pothole_input_name = self.pothole_session.get_inputs()[0].name
        self.pothole_output_names = [output.name for output in self.pothole_session.get_outputs()]
        
        model_input_shape = self.pothole_session.get_inputs()[0].shape
        if len(model_input_shape) == 4 and isinstance(model_input_shape[2], int):
            self.input_size = model_input_shape[2]

        self.total_potholes = 0
        self.frame_count = 0
        
        # Image Enhancement State
        self.prev_frame = None
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            # TRICK 1: Pre-Sharpening (Thickens details before shrinking)
            kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
            img = cv2.filter2D(img, -1, kernel)
            # TRICK 2: INTER_AREA (Preserves pixels during downscale)
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        # TRICK 3: Letterboxing (Preserves Aspect Ratio)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    def preprocess_pothole(self, image):
        img_padded, ratio, (dw, dh) = self.letterbox(image, new_shape=self.input_size)
        rgb_image = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        input_image = rgb_image.astype(np.float32) / 255.0
        input_image = np.transpose(input_image, (2, 0, 1))
        input_image = np.expand_dims(input_image, axis=0)
        return input_image, ratio, dw, dh

    def postprocess_pothole(self, outputs, ratio, dw, dh, original_shape):
        predictions = outputs[0]
        if len(predictions.shape) == 3: predictions = predictions[0]
        if predictions.shape[0] < predictions.shape[1]: predictions = predictions.T

        boxes, scores, class_ids = [], [], []
        for pred in predictions:
            x_center, y_center, width, height = pred[:4]
            conf = np.max(pred[4:])
            cls_id = np.argmax(pred[4:])

            if conf >= self.conf_threshold:
                # Scale back to 1080p coordinates
                x1 = (x_center - width / 2 - dw) / ratio
                y1 = (y_center - height / 2 - dh) / ratio
                x2 = (x_center + width / 2 - dw) / ratio
                y2 = (y_center + height / 2 - dh) / ratio

                x1 = max(0, min(x1, original_shape[1]))
                y1 = max(0, min(y1, original_shape[0]))
                x2 = max(0, min(x2, original_shape[1]))
                y2 = max(0, min(y2, original_shape[0]))

                boxes.append([x1, y1, x2, y2])
                scores.append(float(conf))
                class_ids.append(int(cls_id))

        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
            detections = []
            if len(indices) > 0:
                indices_list = indices.flatten() if hasattr(indices, 'flatten') else indices
                for i in indices_list:
                    detections.append([boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], scores[i], class_ids[i]])
            return detections
        return []

    def process_video(self, video_source, output_path=None):
        if self.camera_mode:
            cap = VideoStream(src=0).start()
            time.sleep(1.0)
        else:
            cap = cv2.VideoCapture(video_source)
            time.sleep(1.0)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS) if not self.camera_mode else 30
        if original_fps <= 0: original_fps = 30

        target_fps = 5.0
        frame_skip = max(1, int(original_fps / target_fps))

        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, int(target_fps), (frame_width, frame_height))

        print("[INFO] Starting Ultimate Pipeline with Image Compression Tricks...")
        frame_counter = 0

        try:
            while True:
                if self.camera_mode:
                    ret, frame = cap.read()
                else:
                    ret, frame = cap.read()
                
                if not ret or frame is None:
                    break

                frame_counter += 1
                if frame_counter % frame_skip != 0:
                    continue

                self.frame_count += 1
                display_frame = frame.copy()

                # TRICK 6: Temporal Blending (Zero-Cost Denoising)
                if self.prev_frame is not None:
                    clean_frame = cv2.addWeighted(frame, 0.8, self.prev_frame, 0.2, 0)
                else:
                    clean_frame = frame.copy()
                self.prev_frame = clean_frame.copy()

                # TRICK 7: Adaptive CLAHE (Shadow/Glare Fixing)
                lab = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
                if np.mean(l_channel) < 100 or np.mean(l_channel) > 200:
                    l_channel = self.clahe.apply(l_channel)
                    lab = cv2.merge((l_channel, a_channel, b_channel))
                    clean_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

                # Pothole Detection (Includes Sharpening, INTER_AREA, and Letterboxing)
                input_tensor, ratio, dw, dh = self.preprocess_pothole(clean_frame)
                pothole_outputs = self.pothole_session.run(self.pothole_output_names, {self.pothole_input_name: input_tensor})
                potholes = self.postprocess_pothole(pothole_outputs, ratio, dw, dh, frame.shape)

                self.total_potholes += len(potholes)

                # Draw
                for det in potholes:
                    x1, y1, x2, y2, conf, cls = det
                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(display_frame, f"Pothole {conf:.2f}", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                if video_writer:
                    video_writer.write(display_frame)

                if self.frame_count % 10 == 0:
                    print(f"[INFO] Frame {self.frame_count} | Potholes: {self.total_potholes}")

        except KeyboardInterrupt:
            pass
        finally:
            if self.camera_mode: cap.stop()
            else: cap.release()
            if video_writer: video_writer.release()
            print(f"\n[INFO] Done. Total Potholes: {self.total_potholes}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pothole-model', required=True)
    parser.add_argument('--video', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    detector = UltimateDetector(pothole_model_path=args.pothole_model)
    detector.process_video(video_source=args.video, output_path=args.output)
