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
DELAY_SECONDS = 20  # khoảng cách tối thiểu giữa 2 lần lưu (giây)

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
    images, classNames = [], []
    myList = os.listdir(path)
    for cl in myList:
        curImg = cv2.imread(os.path.join(path, cl))
        if curImg is not None:
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])
    print(f"✅ Đã tải {len(images)} khuôn mặt: {classNames}")

    encodeListKnown = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img_rgb)
        if len(encodes) > 0:
            encodeListKnown.append(encodes[0])
    print("✅ Mã hoá khuôn mặt hoàn tất.")

    cap = cv2.VideoCapture(0)
    print("🎥 Camera đang mở. Nhấn Q để thoát.")

    threshold = 0.45  # càng thấp thì càng khắt khe (0.4–0.5 là hợp lý)

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
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            if len(faceDis) > 0:
                matchIndex = np.argmin(faceDis)
                best_match_distance = faceDis[matchIndex]

                if best_match_distance < threshold:
                    name = classNames[matchIndex]
                    color = (0, 255, 0)  
                    mark_attendance(name)
                else:
                    name = "Khong xac dinh"
                    color = (0, 0, 255)  

                # Vẽ khung và ghi tên
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
                cv2.putText(img, name.upper(), (x1 + 6, y2 - 6),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Face Attendance', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Đã đóng camera.")
