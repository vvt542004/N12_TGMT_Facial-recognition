import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import os
import cv2
from datetime import datetime
from face_recognition_attendance import start_attendance

# ===============================
# 📘 Hàm tải dữ liệu CSV
# ===============================
def load_attendance_data():
    if os.path.exists('attendance.csv') and os.path.getsize('attendance.csv') > 0:
        try:
            df = pd.read_csv('attendance.csv')
            return df
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=["Name", "Date", "Time"])
    else:
        return pd.DataFrame(columns=["Name", "Date", "Time"])

# ===============================
# 📗 Cập nhật bảng danh sách
# ===============================
def update_table():
    for item in tree.get_children():
        tree.delete(item)
    df = load_attendance_data()
    for _, row in df.iterrows():
        tree.insert("", tk.END, values=(row["Name"], row["Date"], row["Time"]))

# ===============================
# ♻️ Xóa lịch sử (gọi khi bấm "Làm mới danh sách")
# ===============================
def clear_history():
    # Hỏi xác nhận người dùng
    confirm = messagebox.askyesno("Xác nhận xóa", 
                                  "Bạn có chắc muốn xoá toàn bộ lịch sử điểm danh không?\n\n"
                                  "Hành động này không thể hoàn tác.\n\n"
                                  "Nếu muốn lưu trước, hãy bấm 'Xuất File CSV'.")
    if not confirm:
        return

    file = 'attendance.csv'
    try:
        # Option 1: ghi file trống với header
        df_empty = pd.DataFrame(columns=["Name", "Date", "Time"])
        df_empty.to_csv(file, index=False)
        # Nếu bạn muốn hoàn toàn xoá file thay vì ghi trống, bạn có thể dùng:
        # if os.path.exists(file): os.remove(file)
        messagebox.showinfo("Hoàn tất", "Đã xóa lịch sử điểm danh.")
    except Exception as e:
        messagebox.showerror("Lỗi", f"Xóa lịch sử thất bại:\n{e}")

    # Cập nhật lại bảng trên giao diện
    update_table()

# ===============================
# 📷 Mở camera điểm danh
# ===============================
def start_camera():
    messagebox.showinfo("Điểm danh", "Camera đang mở, nhấn Q để thoát.")
    start_attendance()
    update_table()

# ===============================
# 💾 Xuất file CSV sao lưu
# ===============================
def export_csv():
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_file = f"attendance_backup_{now}.csv"
    try:
        df = load_attendance_data()
        df.to_csv(new_file, index=False)
        messagebox.showinfo("Thành công", f"Đã sao lưu file: {new_file}")
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể xuất file CSV:\n{e}")

# ===============================
# 🧍 Thêm khuôn mặt mới vào dataset
# ===============================
def register_new_face():
    name = entry_name.get().strip()
    if not name:
        messagebox.showwarning("Thiếu thông tin", "Vui lòng nhập tên trước khi thêm.")
        return

    save_path = os.path.join("dataset", f"{name}.jpg")
    os.makedirs("dataset", exist_ok=True)

    cap = cv2.VideoCapture(0)
    messagebox.showinfo("Chụp ảnh", "Nhấn phím [S] để chụp và lưu ảnh, [Q] để hủy.")

    while True:
        success, img = cap.read()
        if not success:
            messagebox.showerror("Lỗi camera", "Không thể mở camera.")
            break

        cv2.imshow("Đăng ký khuôn mặt mới", img)
        key = cv2.waitKey(1) & 0xFF

        # Nhấn S để lưu ảnh
        if key == ord('s'):
            cv2.imwrite(save_path, img)
            messagebox.showinfo("Thành công", f"Đã lưu ảnh mới: {save_path}\nKhuôn mặt này sẽ được nhận trong lần điểm danh kế tiếp.")
            break

        # Nhấn Q để thoát
        elif key == ord('q'):
            messagebox.showinfo("Hủy", "Đã hủy đăng ký khuôn mặt.")
            break

    cap.release()
    cv2.destroyAllWindows()

# ===============================
# 🌈 Giao diện chính
# ===============================
root = tk.Tk()
root.title("📸 Hệ thống điểm danh bằng khuôn mặt")
root.geometry("850x650")
root.configure(bg="#eaf4fc")

# ===============================
# 🎯 Tiêu đề chính
# ===============================
title_label = tk.Label(root, text="HỆ THỐNG ĐIỂM DANH BẰNG KHUÔN MẶT",
                       bg="#1a73e8", fg="white",
                       font=("Segoe UI", 16, "bold"), pady=10)
title_label.pack(fill=tk.X)

# ===============================
# 🧍‍♂️ Khu vực thêm người mới
# ===============================
frame_add = tk.LabelFrame(root, text="➕ Thêm người mới", bg="#eaf4fc",
                          font=("Segoe UI", 12, "bold"), padx=15, pady=10)
frame_add.pack(padx=20, pady=10, fill=tk.X)

lbl_name = tk.Label(frame_add, text="Tên:", bg="#eaf4fc", font=("Segoe UI", 11))
lbl_name.grid(row=0, column=0, padx=10)

entry_name = ttk.Entry(frame_add, width=30, font=("Segoe UI", 11))
entry_name.grid(row=0, column=1, padx=10)

btn_register = ttk.Button(frame_add, text="📸 Thêm khuôn mặt", command=register_new_face)
btn_register.grid(row=0, column=2, padx=10)

# ===============================
# 🔘 Các nút chức năng chính
# ===============================
button_frame = tk.Frame(root, bg="#eaf4fc")
button_frame.pack(pady=15)

style = ttk.Style()
style.configure("TButton", font=("Segoe UI", 11), padding=6)

btn_start = ttk.Button(button_frame, text="▶ Mở Camera Điểm Danh", command=start_camera)
btn_start.grid(row=0, column=0, padx=10)

# NOTE: nút "Làm mới danh sách" giờ gọi clear_history (xóa lịch sử).
btn_refresh = ttk.Button(button_frame, text="🔄 Làm mới danh sách", command=clear_history)
btn_refresh.grid(row=0, column=1, padx=10)

btn_export = ttk.Button(button_frame, text="💾 Xuất File CSV", command=export_csv)
btn_export.grid(row=0, column=2, padx=10)

# ===============================
# 🧾 Bảng danh sách điểm danh
# ===============================
frame_table = tk.Frame(root, bg="#eaf4fc")
frame_table.pack(pady=10, fill=tk.BOTH, expand=True)

columns = ("Name", "Date", "Time")
tree = ttk.Treeview(frame_table, columns=columns, show="headings", height=12)
tree.heading("Name", text="Tên")
tree.heading("Date", text="Ngày")
tree.heading("Time", text="Giờ điểm danh")

# Tăng độ rộng và căn giữa
tree.column("Name", width=200, anchor="center")
tree.column("Date", width=150, anchor="center")
tree.column("Time", width=150, anchor="center")

# Thanh cuộn
scrollbar = ttk.Scrollbar(frame_table, orient=tk.VERTICAL, command=tree.yview)
tree.configure(yscroll=scrollbar.set)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
tree.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

# ===============================
# 📅 Footer
# ===============================
footer_label = tk.Label(root, text="© 2025 Nhóm 1 - Face Attendance System",
                        bg="#eaf4fc", fg="#555", font=("Segoe UI", 10))
footer_label.pack(pady=5)

# ===============================
# 🚀 Chạy ứng dụng
# ===============================
update_table()
root.mainloop()
