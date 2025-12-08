import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import os
from datetime import datetime
from face_recognition_attendance import start_attendance

#  Cấu hình hệ thống
ADMIN_PASSWORD = "admin123"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

subjects = ["Xử lý ngôn ngữ tự nhiên", "Hệ nhúng", "Thị giác máy tính", "Lập trình song song"]
MODEL_DIR = "face_models_facenet"
EMBEDDINGS_NPZ = os.path.join(MODEL_DIR, "faces_embeddings_facenet.npz")

ACTIVE_SESSIONS = {s: {"active": False, "start": None, "end": None} for s in subjects}

#  Đọc danh sách sinh viên từ embeddings

student_names = []
if os.path.exists(EMBEDDINGS_NPZ):
    try:
        npz = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
        labels = np.unique(npz["labels"])
        student_names = sorted(labels.tolist())
        print("📂 Danh sách sinh viên đã tải:")
        for name in student_names:
            print("   -", name)
    except Exception as e:
        print("⚠️ Không thể tải danh sách sinh viên:", e)
else:
    print("⚠️ Không tìm thấy file embeddings.")

#  Ứng dụng chính (1 cửa sổ, nhiều frame)

class AttendanceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hệ thống điểm danh bằng khuôn mặt")
        self.geometry("900x600")
        self.resizable(False, False)

        container = tk.Frame(self)
        container.pack(fill="both", expand=True)

        self.frames = {}
        for F in (LoginFrame, AdminFrame, StudentFrame):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(LoginFrame)

    def show_frame(self, cont):
        self.frames[cont].tkraise()


#  Giao diện đăng nhập
class LoginFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Label(self, text="HỆ THỐNG ĐIỂM DANH", font=("Segoe UI", 16, "bold")).pack(pady=30)
        tk.Label(self, text="Chọn vai trò:", font=("Segoe UI", 12)).pack(pady=10)

        ttk.Button(self, text="👨‍💼 Quản trị viên", width=25,
                   command=self.open_admin_login).pack(pady=5)
        ttk.Button(self, text="🎓 Sinh viên", width=25,
                   command=lambda: controller.show_frame(StudentFrame)).pack(pady=5)

    def open_admin_login(self):
        win = tk.Toplevel(self)
        win.title("Đăng nhập Quản trị viên")
        win.geometry("300x180")

        tk.Label(win, text="Nhập mật khẩu:", font=("Segoe UI", 11)).pack(pady=10)
        pw_entry = ttk.Entry(win, show="*", width=25)
        pw_entry.pack(pady=5)

        def verify():
            if pw_entry.get() == ADMIN_PASSWORD:
                win.destroy()
                self.controller.show_frame(AdminFrame)
            else:
                messagebox.showerror("Lỗi", "Sai mật khẩu!")

        ttk.Button(win, text="Đăng nhập", command=verify).pack(pady=10)



#  Giao diện Quản trị viên

class AdminFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Label(self, text="📋 GIAO DIỆN QUẢN TRỊ VIÊN", font=("Segoe UI", 14, "bold"),
                 bg="#1a73e8", fg="white", pady=10).pack(fill=tk.X)

        frame_top = tk.LabelFrame(self, text="Thiết lập buổi học", padx=15, pady=10)
        frame_top.pack(padx=20, pady=10, fill=tk.X)

        # Chọn môn học
        tk.Label(frame_top, text="Chọn môn học:", font=("Segoe UI", 11)).grid(row=0, column=0, padx=10, pady=5)
        self.subject_var = tk.StringVar()
        self.subject_combo = ttk.Combobox(frame_top, textvariable=self.subject_var, values=subjects, width=30)
        self.subject_combo.grid(row=0, column=1, padx=10)

        # Giờ bắt đầu – kết thúc
        tk.Label(frame_top, text="Giờ bắt đầu:", font=("Segoe UI", 11)).grid(row=1, column=0, padx=10, pady=5)
        self.start_time = ttk.Entry(frame_top, width=10)
        self.start_time.insert(0, "07:30")
        self.start_time.grid(row=1, column=1, sticky="w")

        tk.Label(frame_top, text="Giờ kết thúc:", font=("Segoe UI", 11)).grid(row=2, column=0, padx=10, pady=5)
        self.end_time = ttk.Entry(frame_top, width=10)
        self.end_time.insert(0, "09:00")
        self.end_time.grid(row=2, column=1, sticky="w")

        # Các nút điều khiển
        ttk.Button(frame_top, text="🚀 Bắt đầu buổi học", command=self.start_session).grid(row=3, column=0, pady=10)
        ttk.Button(frame_top, text="📄 Xem điểm danh", command=self.load_attendance).grid(row=3, column=1, pady=10)
        ttk.Button(frame_top, text="💾 Xuất file CSV", command=self.export_csv).grid(row=3, column=2, pady=10)
        ttk.Button(frame_top, text="🗑️ Xóa lịch sử điểm danh", command=self.delete_attendance).grid(row=3, column=3, pady=10)
        ttk.Button(frame_top, text="↩️ Quay lại đăng nhập",
                   command=lambda: controller.show_frame(LoginFrame)).grid(row=3, column=4, pady=10)

        self.status_label = tk.Label(self, text="⛔ Chưa có buổi học nào được mở", fg="red", font=("Segoe UI", 11))
        self.status_label.pack(pady=5)

        # Bảng hiển thị danh sách điểm danh
        frame_table = tk.Frame(self)
        frame_table.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        columns = ("Name", "StudentID", "Subject", "Date", "Time")
        self.tree = ttk.Treeview(frame_table, columns=columns, show="headings", height=15)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150, anchor="center")

        scrollbar = ttk.Scrollbar(frame_table, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)

    # ====== Bắt đầu buổi học ======
    def start_session(self):
        subject = self.subject_var.get()
        start = self.start_time.get()
        end = self.end_time.get()

        if not subject:
            messagebox.showwarning("Thiếu thông tin", "Vui lòng chọn môn học trước.")
            return

        ACTIVE_SESSIONS[subject] = {"active": True, "start": start, "end": end}
        self.status_label.config(text=f"✅ Môn '{subject}' đã mở ({start} - {end})", fg="green")
        messagebox.showinfo("Thành công", f"Buổi học '{subject}' đã được mở. Sinh viên có thể điểm danh.")

    # ====== Xem danh sách điểm danh ======
    def load_attendance(self):
        subject = self.subject_var.get()
        today = datetime.now().strftime("%Y-%m-%d")
        path = os.path.join(LOG_DIR, f"log_{subject}_{today}.csv")

        if not os.path.exists(path):
            messagebox.showinfo("Thông báo", "Chưa có dữ liệu điểm danh cho hôm nay.")
            return

        df = pd.read_csv(path)
        for item in self.tree.get_children():
            self.tree.delete(item)
        for _, row in df.iterrows():
            self.tree.insert("", tk.END, values=tuple(row))

    # ====== Xuất file CSV ======
    def export_csv(self):
        subject = self.subject_var.get()
        today = datetime.now().strftime("%Y-%m-%d")
        src = os.path.join(LOG_DIR, f"log_{subject}_{today}.csv")
        if not os.path.exists(src):
            messagebox.showwarning("Lỗi", "Không có dữ liệu để xuất.")
            return
        dest = f"backup_{subject}_{today}.csv"
        os.system(f'copy "{src}" "{dest}"')
        messagebox.showinfo("Thành công", f"Đã xuất file: {dest}")

    # ====== Xóa lịch sử điểm danh ======
    def delete_attendance(self):
        subject = self.subject_var.get()
        if not subject:
            messagebox.showwarning("Thiếu thông tin", "Vui lòng chọn môn học để xóa lịch sử.")
            return

        today = datetime.now().strftime("%Y-%m-%d")
        today_path = os.path.join(LOG_DIR, f"log_{subject}_{today}.csv")

        choice = messagebox.askquestion(
            "Xóa lịch sử",
            f"Chọn chế độ xóa cho môn '{subject}':\n\n"
            f"Yes → Xóa lịch sử của HÔM NAY\nNo → Xóa TOÀN BỘ lịch sử môn này\nCancel → Thoát",
            icon="warning"
        )

        if choice == "yes":
            if os.path.exists(today_path):
                os.remove(today_path)
                for item in self.tree.get_children():
                    self.tree.delete(item)
                messagebox.showinfo("Đã xóa", f"🗑️ Đã xóa lịch sử điểm danh hôm nay của môn '{subject}'.")
            else:
                messagebox.showinfo("Thông báo", f"Không có dữ liệu điểm danh hôm nay của '{subject}'.")

        elif choice == "no":
            deleted = 0
            for file in os.listdir(LOG_DIR):
                if file.startswith(f"log_{subject}_") and file.endswith(".csv"):
                    os.remove(os.path.join(LOG_DIR, file))
                    deleted += 1
            for item in self.tree.get_children():
                self.tree.delete(item)
            messagebox.showinfo("Đã xóa", f"🗑️ Đã xóa {deleted} file lịch sử điểm danh của môn '{subject}'.")
        else:
            return

# 🎓 Giao diện sinh viên (có combobox chọn tên)

class StudentFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Label(self, text="🎓 GIAO DIỆN SINH VIÊN", font=("Segoe UI", 14, "bold"),
                 bg="#1a73e8", fg="white", pady=10).pack(fill=tk.X)

        frame = tk.LabelFrame(self, text="Thông tin sinh viên", padx=20, pady=20)
        frame.pack(padx=20, pady=20, fill=tk.X)

        # combobox chọn tên sinh viên từ embeddings
        tk.Label(frame, text="Họ và tên:", font=("Segoe UI", 11)).grid(row=0, column=0, sticky="w")
        self.name_var = tk.StringVar()
        self.name_combo = ttk.Combobox(frame, textvariable=self.name_var, values=student_names,
                                       width=35, state="readonly")
        self.name_combo.grid(row=0, column=1, pady=5)

        tk.Label(frame, text="Mã SV:", font=("Segoe UI", 11)).grid(row=1, column=0, sticky="w")
        self.id_entry = ttk.Entry(frame, width=35)
        self.id_entry.grid(row=1, column=1, pady=5)

        tk.Label(frame, text="Môn học:", font=("Segoe UI", 11)).grid(row=2, column=0, sticky="w")
        self.subject_var = tk.StringVar()
        self.subject_combo = ttk.Combobox(frame, textvariable=self.subject_var, values=subjects,
                                          width=32, state="readonly")
        self.subject_combo.grid(row=2, column=1, pady=5)

        ttk.Button(self, text="📸 Điểm danh", command=self.start_face_recognition).pack(pady=15)
        ttk.Button(self, text="↩️ Quay lại đăng nhập",
                   command=lambda: controller.show_frame(LoginFrame)).pack(pady=5)

    # ====== Hàm điểm danh ======
    def start_face_recognition(self):
        subject = self.subject_var.get()
        name = self.name_var.get().strip()
        student_id = self.id_entry.get().strip()

        if not all([subject, name, student_id]):
            messagebox.showwarning("Thiếu thông tin", "Vui lòng nhập đầy đủ thông tin và chọn môn học.")
            return

        session_info = ACTIVE_SESSIONS.get(subject, {"active": False, "start": None, "end": None})
        if not session_info["active"]:
            messagebox.showerror("⛔ Chưa đến thời gian", f"Môn '{subject}' chưa được mở để điểm danh.")
            return

        #  Kiểm tra giờ hiện tại so với giờ học
        now_time = datetime.now().strftime("%H:%M")
        start_time = session_info["start"]
        end_time = session_info["end"]

        if now_time < start_time:
            messagebox.showwarning("⏰ Chưa tới giờ điểm danh",
                                   f"Buổi học '{subject}' bắt đầu lúc {start_time}. Vui lòng chờ đến giờ.")
            return

        if now_time > end_time:
            messagebox.showerror("⛔ Quá giờ điểm danh",
                                 f"Buổi học '{subject}' đã kết thúc lúc {end_time}. Không thể điểm danh nữa.")
            return

        messagebox.showinfo("Điểm danh", f"Camera đang mở cho môn {subject}. Nhấn Q để thoát.")
        recognized_name = start_attendance()

        if recognized_name == "unknown" or recognized_name == "Không xác định":
            messagebox.showerror("❌ Thất bại", "Không nhận diện được khuôn mặt hợp lệ.")
            return

        if recognized_name.lower() != name.lower():
            messagebox.showwarning("Sai tên",
                                   f"Khuôn mặt nhận diện là '{recognized_name}', "
                                   f"nhưng bạn chọn '{name}'. Vui lòng chọn đúng tên đã đăng ký.")
            return

        now = datetime.now()
        date, time = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
        log_path = os.path.join(LOG_DIR, f"log_{subject}_{date}.csv")

        df = pd.DataFrame([[name, student_id, subject, date, time]],
                          columns=["Name", "StudentID", "Subject", "Date", "Time"])
        if os.path.exists(log_path):
            old_df = pd.read_csv(log_path)
            df = pd.concat([old_df, df], ignore_index=True)
        df.to_csv(log_path, index=False)

        messagebox.showinfo("Thành công", f"✅ {name} ({student_id}) đã điểm danh môn {subject} thành công!")



if __name__ == "__main__":
    app = AttendanceApp()
    app.mainloop()
