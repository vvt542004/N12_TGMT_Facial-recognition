import cv2
import face_recognition
import numpy as np
import os
import pandas as pd
from datetime import datetime

# ===============================
# ⚙️ Ghi lịch sử điểm danh (chống spam)
# ===============================
last_mark_times = {}
DELAY_SECONDS = 30  # Khoảng cách tối thiểu giữa 2 lần lưu (giây)

def mark_attendance(name):
    """Ghi lịch sử điểm danh vào file CSV"""
    global last_mark_times

    now = datetime.now()
    date = now.strftime('%Y-%m-%d')
    time = now.strftime('%H:%M:%S')

    # Chống spam trong thời gian ngắn
    if name in last_mark_times:
        delta = (now - last_mark_times[name]).total_seconds()
        if delta < DELAY_SECONDS:
            return
    last_mark_times[name] = now

    file = 'attendance.csv'
    if not os.path.exists(file) or os.path.getsize(file) == 0:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_csv(file, index=False)

    try:
        df = pd.read_csv(file)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    new_row = pd.DataFrame([[name, date, time]], columns=["Name", "Date", "Time"])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(file, index=False)

    print(f"✅ Đã lưu điểm danh: {name} ({date} {time})")

# ===============================
# 🎥 Nhận diện & điểm danh
# ===============================
def start_attendance():
    path = 'dataset'
    if not os.path.exists(path) or len(os.listdir(path)) == 0:
        print("⚠ Thư mục dataset trống. Hãy thêm khuôn mặt trước.")
        return

    print("📂 Đang tải dataset...")
    encode_dict = {}  # { 'tuan': [encode1, encode2, ...], ... }

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        name = os.path.splitext(filename)[0].split('_')[0]  # tách tên trước dấu _
        img = cv2.imread(file_path)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(rgb)
        if len(encodes) > 0:
            encode_dict.setdefault(name, []).append(encodes[0])

    print(f"✅ Đã tải {len(encode_dict)} người: {list(encode_dict.keys())}")

    cap = cv2.VideoCapture(0)
    print("🎥 Camera đang mở. Nhấn Q để thoát.")

    threshold = 0.35 # càng thấp thì càng khắt khe

    while True:
        success, img = cap.read()
        if not success:
            print("❌ Không thể mở camera.")
            break

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            best_name = "Không xác định"
            min_distance = 1.0

            # So sánh với từng người
            for name, encodes in encode_dict.items():
                distances = face_recognition.face_distance(encodes, encodeFace)
                avg_distance = np.mean(distances)
                if avg_distance < min_distance:
                    min_distance = avg_distance
                    best_name = name

            # Kiểm tra ngưỡng
            if min_distance < threshold:
                color = (0, 255, 0)
                mark_attendance(best_name)
            else:
                best_name = "Không xác định"
                color = (0, 0, 255)

            # Vẽ khung
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
            cv2.putText(img, best_name.upper(), (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Face Attendance', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Đã đóng camera.")
