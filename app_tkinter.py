import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import os
from face_recognition_attendance import start_attendance
from datetime import datetime

# ===============================
# 📘 Hàm tải dữ liệu CSV
# ===============================
def load_attendance_data():
    if os.path.exists('attendance.csv') and os.path.getsize('attendance.csv') > 0:
        df = pd.read_csv('attendance.csv')
        return df
    else:
        messagebox.showinfo("Thông báo", "Chưa có dữ liệu điểm danh.")
        return pd.DataFrame(columns=["Name", "Date", "Time"])

# ===============================
# 📗 Hàm hiển thị dữ liệu vào bảng
# ===============================
def update_table():
    for item in tree.get_children():
        tree.delete(item)
    df = load_attendance_data()
    for _, row in df.iterrows():
        tree.insert("", tk.END, values=(row["Name"], row["Date"], row["Time"]))

# ===============================
# 📷 Hàm mở camera điểm danh
# ===============================
def start_camera():
    messagebox.showinfo("Điểm danh", "Camera đang mở, nhấn Q để thoát.")
    start_attendance()
    update_table()

# ===============================
# 💾 Hàm xuất file CSV (sao lưu)
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
# 🌈 Giao diện chính
# ===============================
root = tk.Tk()
root.title("📸 Hệ thống điểm danh bằng khuôn mặt")
root.geometry("800x600")
root.configure(bg="#eaf4fc")

# ===============================
# 🎯 Tiêu đề chính
# ===============================
title_label = tk.Label(root, text="HỆ THỐNG ĐIỂM DANH BẰNG KHUÔN MẶT",
                       bg="#1a73e8", fg="white",
                       font=("Segoe UI", 16, "bold"), pady=10)
title_label.pack(fill=tk.X)

# ===============================
# 🔘 Các nút chức năng
# ===============================
button_frame = tk.Frame(root, bg="#eaf4fc")
button_frame.pack(pady=15)

style = ttk.Style()
style.configure("TButton", font=("Segoe UI", 11), padding=6)

btn_start = ttk.Button(button_frame, text="▶ Mở Camera Điểm Danh", command=start_camera)
btn_start.grid(row=0, column=0, padx=10)

btn_refresh = ttk.Button(button_frame, text="🔄 Làm mới danh sách", command=update_table)
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
