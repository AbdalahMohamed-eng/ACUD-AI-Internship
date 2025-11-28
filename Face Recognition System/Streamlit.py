import os
import cv2
import numpy as np
import joblib
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
from collections import deque
import tensorflow as tf
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder, Normalizer, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import time
from mtcnn import MTCNN
import av

# ================= Configuration =================
SVM_PATH = r"C:\Users\hp\project\svm_model.pkl"
RF_PATH = r"C:\Users\hp\project\rf_model.pkl"
ENCODER_PATH = r"C:\Users\hp\project\encoder.pkl"
SCALER_PATH = r"C:\Users\hp\project\scaler.pkl"
YOLO_MODEL_PATH = r"C:\Users\hp\project\yolov8n.pt"

IMG_SIZE = (160, 160)
MARGIN = 50
UNKNOWN_THRESHOLD = 0.6
MIN_QUEUE_SIZE = 3
MAX_FACES = 3
FRAME_SKIP = 1
FRAME_SIZE = (640, 480)

# ================= Optional CPU/thread fixes =================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

# ================= Load models =================
@st.cache_resource
def load_models():
    detector = MTCNN()
    embedder = FaceNet()
    svm = joblib.load(SVM_PATH)
    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    l2 = Normalizer(norm="l2")
    rf = joblib.load(RF_PATH)
    yolo_model = YOLO(YOLO_MODEL_PATH)
    return detector, embedder, svm, encoder, scaler, l2, rf, yolo_model

detector, embedder, svm, encoder, scaler, l2, rf, yolo_model = load_models()

# ================= Utility Functions =================
def square_crop_with_margin(box, img_w, img_h, margin=MARGIN):
    x, y, w, h = box
    x, y = abs(x), abs(y)
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(img_w, x + w + margin), min(img_h, y + h + margin)
    bw, bh = x2 - x1, y2 - y1
    side = max(bw, bh)
    cx = x1 + bw // 2
    cy = y1 + bh // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(img_w, x1 + side)
    y2 = min(img_h, y1 + side)
    return x1, y1, x2, y2

def align_face(image, box, landmarks=None):
    if landmarks is not None:
        left_eye = np.array(landmarks[0][0])
        right_eye = np.array(landmarks[1][0])
        angle = np.arctan2(
            right_eye[1] - left_eye[1],
            right_eye[0] - left_eye[0]
        ) * 180 / np.pi
        M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, 1.0)
        aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        x1, y1, x2, y2 = square_crop_with_margin(box, image.shape[1], image.shape[0])
        return aligned[y1:y2, x1:x2]
    else:
        x1, y1, x2, y2 = square_crop_with_margin(box, image.shape[1], image.shape[0])
        return image[y1:y2, x1:x2]
    
def face_to_embedding(face_rgb_01):
    face_255 = (face_rgb_01 * 255).astype(np.uint8)
    face_255 = np.expand_dims(face_255, 0)
    embs = embedder.embeddings(face_255)
    return embs[0]

# ================= Face Recognition Processor =================
class FaceRecognitionProcessor(VideoProcessorBase):
    def __init__(self):
        self.svm = svm
        self.rf = rf
        self.encoder = encoder
        self.scaler = scaler
        self.l2 = l2
        self.detector = detector
        self.pred_queues = {}
        self.frame_count = 0
        self.last_processed_frame = None

    def extract_face(self, img, required_size=IMG_SIZE):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb) if detector else []
        if not results:
            results_yolo = yolo_model(rgb)
            boxes = results_yolo[0].boxes
            if len(boxes) > 0:
                for box in boxes:
                    if int(box.cls) == 0:
                        box_coords = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, box_coords)
                        face = rgb[y1:y2, x1:x2]
                        break
                else:
                    return None
            else:
                return None
        else:
            box = results[0]['box']
            landmarks = [[results[0]['keypoints']['left_eye']], [results[0]['keypoints']['right_eye']]]
            face = align_face(rgb, box, landmarks)
        if face.size == 0:
            return None
        face = cv2.resize(face, required_size)
        face = face.astype("float32") / 255.0
        return face

    def process_image(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb) if detector else []
        faces = []
        boxes = []
        predictions = []
        
        if results:
            for result in results[:MAX_FACES]:
                box = result['box']
                landmarks = [[result['keypoints']['left_eye']], [result['keypoints']['right_eye']]]
                face = align_face(rgb, box, landmarks)
                if face.size == 0:
                    continue
                faces.append((face, box))
                boxes.append(box)
        else:
            results_yolo = yolo_model(rgb)
            for box in results_yolo[0].boxes:
                if int(box.cls) == 0:
                    box_coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = box_coords
                    face = align_face(rgb, [x1, y1, x2 - x1, y2 - y1])
                    if face.size == 0:
                        continue
                    faces.append((face, [x1, y1, x2 - x1, y2 - y1]))
                    boxes.append([x1, y1, x2 - x1, y2 - y1])
        
        for i, (face, box) in enumerate(faces):
            face = cv2.resize(face, IMG_SIZE).astype(np.float32) / 255.0
            emb = face_to_embedding(face).reshape(1, -1)
            emb = self.l2.transform(emb)
            emb = self.scaler.transform(emb)
            
            probs_svm = self.svm.predict_proba(emb)[0]
            probs_rf = self.rf.predict_proba(emb)[0]
            probs = (probs_svm + probs_rf) / 2
            pred_idx = np.argmax(probs)
            pred_conf = probs[pred_idx]
            
            pred_name = self.encoder.inverse_transform([pred_idx])[0]
            display = f"Face {i+1}: {pred_name} ({pred_conf*100:.1f}%)" if pred_conf >= UNKNOWN_THRESHOLD else f"Face {i+1}: Unknown ({pred_conf*100:.1f}%)"
            color = (0, 255, 0) if pred_conf >= UNKNOWN_THRESHOLD else (0, 0, 255)
            predictions.append(display)
            
            x, y, w, h = box
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        
        return img, predictions

    def realtime_recognition(self, frame):
        frame = cv2.resize(frame, FRAME_SIZE)
        self.frame_count += 1

        if self.frame_count % FRAME_SKIP != 0 and self.last_processed_frame is not None:
            return self.last_processed_frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = []
        start_time = time.time()

        try:
            results = self.detector.detect_faces(rgb)
        except Exception as e:
            st.error(f"MTCNN failed: {e}")
            results = []

        if results:
            for result in results[:MAX_FACES]:
                box = result['box']
                landmarks = [[result['keypoints']['left_eye']], [result['keypoints']['right_eye']]]
                face = align_face(rgb, box, landmarks)
                if face.size == 0:
                    continue
                faces.append((face, box))
        else:
            results_yolo = yolo_model(rgb)
            for box in results_yolo[0].boxes:
                if int(box.cls) == 0:
                    box_coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = box_coords
                    face = align_face(rgb, [x1, y1, x2 - x1, y2 - y1])
                    if face.size == 0:
                        continue
                    faces.append((face, [x1, y1, x2 - x1, y2 - y1]))

        for face, box in faces:
            face = cv2.resize(face, IMG_SIZE).astype(np.float32) / 255.0
            emb = face_to_embedding(face).reshape(1, -1)
            emb = self.l2.transform(emb)
            emb = self.scaler.transform(emb)

            probs = self.svm.predict_proba(emb)[0]
            pred_idx = np.argmax(probs)
            pred_conf = probs[pred_idx]

            track_id = f"face_{self.frame_count}"
            if track_id not in self.pred_queues:
                self.pred_queues[track_id] = deque(maxlen=3)
            self.pred_queues[track_id].append((pred_idx, pred_conf))

            if len(self.pred_queues[track_id]) >= MIN_QUEUE_SIZE:
                avg_conf = np.mean([c for _, c in self.pred_queues[track_id]])
                mode_idx = max(
                    set([i for i, _ in self.pred_queues[track_id]]),
                    key=[i for i, _ in self.pred_queues[track_id]].count
                )
                pred_name = self.encoder.inverse_transform([mode_idx])[0]
                display = f"{pred_name} ({avg_conf*100:.1f}%)" if avg_conf >= UNKNOWN_THRESHOLD else f"Unknown ({avg_conf*100:.1f}%)"
                color = (0, 255, 0) if avg_conf >= UNKNOWN_THRESHOLD else (0, 0, 255)
            else:
                pred_name = self.encoder.inverse_transform([pred_idx])[0]
                display = f"{pred_name} ({pred_conf*100:.1f}%)" if pred_conf >= UNKNOWN_THRESHOLD else f"Unknown ({pred_conf*100:.1f}%)"
                color = (0, 255, 0) if pred_conf >= UNKNOWN_THRESHOLD else (0, 0, 255)

            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, display, (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        self.pred_queues = {k: q for k, q in self.pred_queues.items() if k == f"face_{self.frame_count}"}

        end_time = time.time()
        fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        self.last_processed_frame = frame
        return frame

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = self.realtime_recognition(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ================= Streamlit App =================
st.title("Live Face Recognition")
st.write("Choose an option: use webcam for real-time face recognition or upload an image for face detection and recognition.")

# Tabs for webcam and image upload
tab1, tab2 = st.tabs(["Webcam", "Upload Image"])

with tab1:
    st.write("Using webcam feed to detect and recognize faces in real-time. Ensure webcam is enabled.")
    if svm is not None and rf is not None:
        webrtc_streamer(
            key="face-recognition",
            video_processor_factory=FaceRecognitionProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    else:
        st.error("Failed to load models. Please check file paths and try again.")

with tab2:
    st.write("Upload an image to detect and recognize faces.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Process the image
        processor = FaceRecognitionProcessor()
        processed_img, predictions = processor.process_image(img)
        
        # Display the processed image
        st.image(processed_img, channels="BGR", caption="Processed Image with Face Detection")
        
        # Display predictions outside the image
        if predictions:
            st.write("**Face Recognition Results:**")
            for pred in predictions:
                st.write(pred)
        else:
            st.write("No faces detected in the uploaded image.")
    else:
        st.info("Please upload an image to proceed.")