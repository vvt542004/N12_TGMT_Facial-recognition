import os
import cv2
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
from mtcnn import MTCNN
from deepface import DeepFace

MODEL_DIR = "face_models_facenet"
SVM_PATH = os.path.join(MODEL_DIR, "svm_facenet.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder_facenet.pkl")
EMBEDDINGS_NPZ = os.path.join(MODEL_DIR, "faces_embeddings_facenet.npz")

SVM_PROB_THRESH = 0.45
COSINE_SIM_THRESH = 0.5
FRAMES_REQUIRED = 3
DELAY_SECONDS = 30

print(" Đang tải mô hình...")

with open(SVM_PATH, "rb") as f:
    svm_model = pickle.load(f)
with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

print(" Mô hình đã sẵn sàng.")


# Tải embeddings đã lưu

embeddings_by_label = {}
if os.path.exists(EMBEDDINGS_NPZ):
    try:
        npz = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
        X = npz["embeddings"]
        y = npz["labels"]
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
        for emb, lbl in zip(X_norm, y):
            embeddings_by_label.setdefault(lbl, []).append(emb)

        print(f" Đã tải embeddings của {len(embeddings_by_label)} lớp.")
        print(" Danh sách lớp đã tải:")
        for lbl, embs in embeddings_by_label.items():
            print(f"   - {lbl}: {len(embs)} ảnh")

    except Exception as e:
        print(" Không thể tải file embeddings:", e)
else:
    print(" Không tìm thấy file embeddings, chỉ dùng SVM để nhận diện.")

detector = MTCNN()

# Hàm phụ trợ

def l2_normalize(v):
    v = np.array(v)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def mean_cosine_sim(emb, label):
    if label not in embeddings_by_label:
        return 0.0
    arr = np.stack(embeddings_by_label[label], axis=0)
    sims = np.dot(arr, emb)
    return float(np.mean(sims))


#lịch sử điểm danh

last_mark_times = {}

def mark_attendance(name):
    now = datetime.now()
    date = now.strftime('%Y-%m-%d')
    time = now.strftime('%H:%M:%S')

    if name in last_mark_times:
        delta = (now - last_mark_times[name]).total_seconds()
        if delta < DELAY_SECONDS:
            return
    last_mark_times[name] = now

    file = "attendance.csv"
    if not os.path.exists(file) or os.path.getsize(file) == 0:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_csv(file, index=False)

    df = pd.read_csv(file)
    new_row = pd.DataFrame([[name, date, time]], columns=["Name", "Date", "Time"])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(file, index=False)
    print(f" Đã lưu điểm danh: {name} ({date} {time})")


# Nhận diện khuôn mặt
def start_attendance():
    print(" Đang mở camera... (nhấn Q để thoát)")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Không thể mở camera.")
        return "unknown"

    DeepFace.build_model("Facenet")
    recognized_name = "unknown"
    frame_confirm = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)

        for f in faces:
            x, y, w, h = f["box"]
            x, y = max(0, x), max(0, y)
            face = rgb[y:y + h, x:x + w]
            if face.size == 0:
                continue

            try:
                rep = DeepFace.represent(img_path=face, model_name="Facenet", enforce_detection=False)
                emb = l2_normalize(np.array(rep[0]["embedding"]))
            except Exception:
                continue

            probs = svm_model.predict_proba([emb])[0]
            max_prob = float(np.max(probs))
            pred_idx = np.argmax(probs)
            pred_name = label_encoder.inverse_transform([pred_idx])[0]
            avg_sim = mean_cosine_sim(emb, pred_name) if embeddings_by_label else 0.0

            recognized = (max_prob >= SVM_PROB_THRESH) and (avg_sim >= COSINE_SIM_THRESH)
            name_display = pred_name if recognized else "unknown"
            color = (0, 255, 0) if recognized else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame,name_display, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            #  Nếu xác nhận hợp lệ qua nhiều frame → điểm danh + tắt camera
            if recognized:
                count, last_time = frame_confirm.get(pred_name, (0, datetime.min))
                if (datetime.now() - last_time).total_seconds() < 2:
                    count += 1
                else:
                    count = 1
                frame_confirm[pred_name] = (count, datetime.now())

                if count >= FRAMES_REQUIRED:
                    recognized_name = pred_name
                    mark_attendance(pred_name)

                    # Hiển thị thông báo điểm danh thành công
                    cv2.putText(frame, f"Diem danh thanh cong: {pred_name}",
                                (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)
                    cv2.imshow("Face Attendance", frame)
                    print(f" Điểm danh thành công cho {pred_name}")

                    cv2.waitKey(1500)
                    cap.release()
                    cv2.destroyAllWindows()
                    return recognized_name
            else:
                recognized_name = "unknown"

        cv2.imshow("Face Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if recognized_name != "unknown":
        return recognized_name
    else:
        print(" Không nhận diện được khuôn mặt hợp lệ.")
        return "unknown"
