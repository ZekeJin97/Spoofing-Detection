import os
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp

# === Setup ===
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)


def extract_faces_from_videos(video_dir, label, out_dir, every_n_frames=15, max_per_video=5, margin=0.3):
    os.makedirs(out_dir, exist_ok=True)
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])

    for video_name in tqdm(video_files, desc=f"Processing {label}"):
        path = os.path.join(video_dir, video_name)
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            print(f"❌ Failed to open {video_name}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_every = max(every_n_frames, total_frames // max_per_video)

        frame_idx = 0
        saved = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every != 0:
                frame_idx += 1
                continue

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = mp_face.process(img_rgb)

            if result.detections:
                face = result.detections[0]
                bbox = face.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x1 = int(max((bbox.xmin - margin * bbox.width) * w, 0))
                y1 = int(max((bbox.ymin - margin * bbox.height) * h, 0))
                x2 = int(min((bbox.xmin + bbox.width * (1 + margin)) * w, w))
                y2 = int(min((bbox.ymin + bbox.height * (1 + margin)) * h, h))

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                # Naming: keep original filename + frame index
                name_no_ext = os.path.splitext(video_name)[0]
                out_path = os.path.join(out_dir, f"{name_no_ext}_frame{frame_idx}.jpg")
                cv2.imwrite(out_path, face_crop)
                saved += 1

            frame_idx += 1
            if saved >= max_per_video:
                break

        cap.release()


# === Usage ===
extract_faces_from_videos("Celeb-real", label="real", out_dir="balanced_dataset/train/real", every_n_frames=15)
extract_faces_from_videos("Celeb-synthesis", label="fake", out_dir="balanced_dataset/train/fake_generated", every_n_frames=15)
