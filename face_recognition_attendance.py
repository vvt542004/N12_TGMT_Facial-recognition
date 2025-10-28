import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# ==============================
# 1️⃣ Hàm mã hóa khuôn mặt
# ==============================
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if len(encodes) > 0:
            encodeList.append(encodes[0])
        else:
            print("⚠️ Không phát hiện được khuôn mặt trong một ảnh.")
    return encodeList


# ==============================
# 2️⃣ Hàm ghi / cập nhật điểm danh
# ==============================
def markAttendance(name):
    # Nếu file chưa tồn tại → tạo mới
    if not os.path.exists('attendance.csv'):
        with open('attendance.csv', 'w', encoding='utf-8') as f:
            f.write('Name,Date,Time\n')

    # Đọc toàn bộ dữ liệu hiện có
    lines = []
    if os.path.getsize('attendance.csv') > 0:
        with open('attendance.csv', 'r', encoding='utf-8') as f:
            lines = f.readlines()

    # Bỏ qua tiêu đề
    header = "Name,Date,Time\n"
    data_lines = [line for line in lines if not line.startswith("Name")]

    # Lấy ngày hiện tại
    now = datetime.now()
    dateString = now.strftime('%Y-%m-%d')
    timeString = now.strftime('%H:%M:%S')

    # Xóa bản ghi cũ của người đó trong cùng ngày (nếu có)
    new_data = []
    for line in data_lines:
        entry = line.strip().split(',')
        if len(entry) >= 2:
            existing_name, existing_date = entry[0], entry[1]
            if existing_name == name and existing_date == dateString:
                continue  # Bỏ dòng cũ cùng ngày

        new_data.append(line)

    # Ghi dòng mới
    new_data.append(f"{name},{dateString},{timeString}\n")

    # Ghi lại toàn bộ file
    with open('attendance.csv', 'w', encoding='utf-8') as f:
        f.write(header)
        f.writelines(new_data)

    print(f'✅ Đã cập nhật điểm danh: {name} - {dateString} {timeString}')


# ==============================
# 3️⃣ Hàm chính mở camera & nhận diện
# ==============================
def start_attendance():
    path = 'dataset'
    images = []
    classNames = []

    # Đọc dữ liệu khuôn mặt mẫu
    if not os.path.exists(path):
        print(f"❌ Không tìm thấy thư mục {path}.")
        return

    myList = os.listdir(path)
    print("Danh sách ảnh trong dataset:", myList)

    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        if curImg is None:
            print(f"⚠️ Không thể đọc ảnh: {cl}")
            continue
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

    # Mã hóa khuôn mặt
    print("Đang mã hóa khuôn mặt, vui lòng chờ...")
    encodeListKnown = findEncodings(images)
    print("✅ Đã mã hóa xong khuôn mặt trong dataset!")

    # Mở camera
    cap = cv2.VideoCapture(0)
    threshold = 0.5  # Ngưỡng xác định độ khớp

    while True:
        success, img = cap.read()
        if not success:
            print("❌ Không thể truy cập camera.")
            break

        # Giảm kích thước để tăng tốc
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            if len(faceDis) == 0:
                continue

            matchIndex = np.argmin(faceDis)
            name = "Unknown"
            color = (0, 0, 255)  # Mặc định khung đỏ

            # Nếu độ khác biệt nhỏ hơn ngưỡng → nhận diện thành công
            if faceDis[matchIndex] < threshold:
                name = classNames[matchIndex].upper()
                color = (0, 255, 0)
                markAttendance(name)

            # Vẽ khung khuôn mặt
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Hiển thị hình ảnh camera
        cv2.imshow('Face Attendance', img)

        # Nhấn Q để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ==============================
# 4️⃣ Cho phép chạy độc lập
# ==============================
if __name__ == "__main__":
    start_attendance()
