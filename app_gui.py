"""
app_gui.py
Giao diện đồ họa nhận diện động vật.
Hỗ trợ 2 mô hình:
  - YOLOv8-cls  (Ultralytics)
  - MobileNetV3 (PyTorch torchvision)
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from PIL import Image, ImageTk
import torch
from torchvision import transforms

BASE_DIR = Path(__file__).parent

# ─── Màu sắc & Font ───────────────────────────────────────────
BG_DARK    = "#1E1E2E"
BG_PANEL   = "#282A36"
BG_CARD    = "#313244"
ACCENT     = "#CBA6F7"   # mauve
GREEN      = "#A6E3A1"
RED        = "#F38BA8"
YELLOW     = "#F9E2AF"
TEXT_MAIN  = "#CDD6F4"
TEXT_MUTED = "#6C7086"
FONT_TITLE = ("Segoe UI", 17, "bold")
FONT_HEAD  = ("Segoe UI", 12, "bold")
FONT_BODY  = ("Segoe UI", 11)
FONT_SMALL = ("Segoe UI", 9)
# ──────────────────────────────────────────────────────────────


class ModelBackend:
    """Lớp trừu tượng cho 2 backend: YOLO và MobileNetV3."""

    def __init__(self, model_type: str):
        self.model_type   = model_type
        self.model        = None
        self.classes      = []
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform    = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.status       = "not_loaded"   # not_loaded | loaded | missing
        self.status_msg   = ""

    # ─── MobileNetV3 ───
    def load_mobilenet(self):
        from torchvision import models
        import torch.nn as nn

        model_path   = BASE_DIR / "best_model.pth"
        classes_path = BASE_DIR / "classes.txt"

        if not model_path.exists() or not classes_path.exists():
            self.status     = "missing"
            self.status_msg = "Chưa có best_model.pth — chạy train.py trước"
            return

        with open(classes_path, "r", encoding="utf-8") as f:
            self.classes = [l.strip() for l in f if l.strip()]

        model = models.mobilenet_v3_large()
        in_f  = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_f, len(self.classes))
        model.load_state_dict(torch.load(str(model_path), map_location=self.device))
        self.model  = model.to(self.device).eval()
        self.status = "loaded"

    # ─── YOLOv8 ───
    def load_yolo(self, variant: str = "n"):
        from ultralytics import YOLO

        suffix = f"yolov8{variant}-cls"
        pt     = BASE_DIR / "runs" / "classify" / suffix / "weights" / "best.pt"

        if not pt.exists():
            self.status     = "missing"
            self.status_msg = f"Chưa có {pt.name} — chạy train_yolo.py trước"
            return

        self.model  = YOLO(str(pt))
        self.status = "loaded"

    # ─── Predict ───
    def predict(self, pil_image: Image.Image):
        if self.status != "loaded":
            return None, 0.0

        if self.model_type.startswith("yolo"):
            results = self.model.predict(
                source=pil_image, imgsz=224, verbose=False
            )
            r         = results[0]
            probs     = r.probs
            top_cls   = r.names[probs.top1]
            top_conf  = float(probs.top1conf)
            return top_cls, top_conf * 100

        else:  # mobilenet
            tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out   = self.model(tensor)
                prob  = torch.nn.functional.softmax(out[0], dim=0)
                conf, idx = prob.max(0)
            return self.classes[idx.item()], conf.item() * 100


class AnimalClassifierApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Nhận Diện Động Vật — YOLO & MobileNetV3")
        self.root.geometry("900x680")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(False, False)

        # State
        self.selected_image: Image.Image | None = None
        self.model_var = tk.StringVar(value="mobilenet")
        self.backend   = ModelBackend("mobilenet")

        self._build_ui()
        self._load_model()

    # ─────────────────────── UI ───────────────────────────────
    def _build_ui(self):
        # ── Header ──
        hdr = tk.Frame(self.root, bg=BG_DARK)
        hdr.pack(fill="x", padx=0, pady=(18, 0))
        tk.Label(hdr, text="🐾  Nhận Diện Động Vật", font=FONT_TITLE,
                 bg=BG_DARK, fg=ACCENT).pack()
        tk.Label(hdr, text="So sánh YOLO & MobileNetV3 — Transfer Learning",
                 font=FONT_SMALL, bg=BG_DARK, fg=TEXT_MUTED).pack(pady=(2, 0))

        # ── Body ──
        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=24, pady=14)

        # Left: Image preview
        left = tk.Frame(body, bg=BG_PANEL, bd=0,
                        highlightthickness=1, highlightbackground="#45475A",
                        width=430, height=390)
        left.pack_propagate(False)
        left.pack(side="left", fill="y")

        self.img_label = tk.Label(left, text="Chọn ảnh để bắt đầu",
                                  font=FONT_BODY, bg=BG_PANEL, fg=TEXT_MUTED)
        self.img_label.pack(expand=True)

        # Right: Controls + Result
        right = tk.Frame(body, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True, padx=(18, 0))

        # Model selector card
        sel_card = tk.LabelFrame(right, text=" Chọn mô hình ",
                                 font=FONT_HEAD, bg=BG_PANEL, fg=ACCENT,
                                 bd=1, relief="solid",
                                 highlightbackground="#45475A")
        sel_card.pack(fill="x", pady=(0, 10))

        choices = [
            ("MobileNetV3-Large (PyTorch)", "mobilenet"),
            ("YOLOv8n-cls  — Nano  (~6 MB)", "yolo_n"),
            ("YOLOv8s-cls  — Small (~22 MB)", "yolo_s"),
        ]
        for text, val in choices:
            rb = tk.Radiobutton(
                sel_card, text=text, variable=self.model_var, value=val,
                font=FONT_BODY, bg=BG_PANEL, fg=TEXT_MAIN,
                selectcolor=BG_DARK, activebackground=BG_PANEL,
                activeforeground=ACCENT, cursor="hand2",
                command=self._on_model_change,
            )
            rb.pack(anchor="w", padx=14, pady=3)

        # Status chip
        self.status_lbl = tk.Label(right, text="", font=FONT_SMALL,
                                   bg=BG_DARK, fg=YELLOW)
        self.status_lbl.pack(anchor="w")

        # Result card
        res_card = tk.Frame(right, bg=BG_CARD,
                            highlightthickness=1, highlightbackground="#45475A")
        res_card.pack(fill="x", pady=(8, 0))

        tk.Label(res_card, text="Kết quả nhận diện", font=FONT_HEAD,
                 bg=BG_CARD, fg=TEXT_MUTED).pack(anchor="w", padx=14, pady=(10, 0))

        self.pred_lbl = tk.Label(res_card, text="—", font=("Segoe UI", 22, "bold"),
                                 bg=BG_CARD, fg=GREEN)
        self.pred_lbl.pack(anchor="w", padx=14, pady=(4, 0))

        self.conf_lbl = tk.Label(res_card, text="", font=FONT_BODY,
                                 bg=BG_CARD, fg=TEXT_MAIN)
        self.conf_lbl.pack(anchor="w", padx=14, pady=(2, 10))

        # Confidence bar
        self.bar_var = tk.DoubleVar(value=0)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("green.Horizontal.TProgressbar",
                        troughcolor=BG_PANEL, background=GREEN,
                        bordercolor=BG_PANEL, lightcolor=GREEN, darkcolor=GREEN)
        self.pbar = ttk.Progressbar(right, length=400, mode="determinate",
                                    variable=self.bar_var,
                                    style="green.Horizontal.TProgressbar")
        self.pbar.pack(fill="x", pady=(6, 0))

        # Buttons
        btn_frame = tk.Frame(right, bg=BG_DARK)
        btn_frame.pack(fill="x", pady=(16, 0))

        self.btn_open = tk.Button(
            btn_frame, text="  📂  Chọn Ảnh", font=FONT_HEAD,
            bg=ACCENT, fg=BG_DARK, activebackground="#B4BEFE",
            bd=0, padx=18, pady=9, cursor="hand2",
            command=self._open_image,
        )
        self.btn_open.pack(side="left")

        self.btn_again = tk.Button(
            btn_frame, text="  🔁  Dự đoán lại", font=FONT_HEAD,
            bg=BG_CARD, fg=TEXT_MAIN, activebackground="#45475A",
            bd=0, padx=14, pady=9, cursor="hand2",
            command=self._predict,
        )
        self.btn_again.pack(side="left", padx=(10, 0))

        # Hover effects
        for btn, normal, hover in [
            (self.btn_open,  ACCENT,   "#B4BEFE"),
            (self.btn_again, BG_CARD,  "#45475A"),
        ]:
            btn.bind("<Enter>", lambda e, b=btn, h=hover: b.config(bg=h))
            btn.bind("<Leave>", lambda e, b=btn, n=normal: b.config(bg=n))

        # Device info
        dev = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        tk.Label(right, text=f"Device: {dev}", font=FONT_SMALL,
                 bg=BG_DARK, fg=TEXT_MUTED).pack(anchor="e", pady=(8, 0))

    # ──────────────────── Logic ───────────────────────────────
    def _on_model_change(self):
        self._load_model()
        if self.selected_image:
            self._predict()

    def _load_model(self):
        val = self.model_var.get()
        self.backend = ModelBackend(val)

        if val == "mobilenet":
            self.backend.load_mobilenet()
        elif val == "yolo_n":
            self.backend.load_yolo("n")
        elif val == "yolo_s":
            self.backend.load_yolo("s")

        if self.backend.status == "loaded":
            self.status_lbl.config(
                text=f"✅ Mô hình {val} đã sẵn sàng", fg=GREEN)
        elif self.backend.status == "missing":
            self.status_lbl.config(
                text=f"⚠️  {self.backend.status_msg}", fg=YELLOW)
        else:
            self.status_lbl.config(text="⏳ Đang tải mô hình...", fg=YELLOW)

    def _open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff")]
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
            self.selected_image = img
            self._show_thumbnail(img)
            self._predict()
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể mở ảnh:\n{e}")

    def _show_thumbnail(self, img: Image.Image):
        thumb = img.copy()
        thumb.thumbnail((428, 388))
        self.photo = ImageTk.PhotoImage(thumb)
        self.img_label.config(image=self.photo, text="")

    def _predict(self):
        if not self.selected_image:
            messagebox.showinfo("Thông báo", "Hãy chọn ảnh trước.")
            return
        if self.backend.status != "loaded":
            messagebox.showwarning("Chưa có mô hình", self.backend.status_msg or
                                   "Mô hình chưa được tải.")
            return

        self.pred_lbl.config(text="Đang nhận diện...", fg=YELLOW)
        self.root.update_idletasks()

        try:
            label, conf = self.backend.predict(self.selected_image)
            self.pred_lbl.config(text=label.upper(), fg=GREEN)
            self.conf_lbl.config(text=f"Độ tin cậy: {conf:.2f}%")
            self.bar_var.set(conf)
        except Exception as e:
            self.pred_lbl.config(text="Lỗi!", fg=RED)
            self.conf_lbl.config(text=str(e))
            self.bar_var.set(0)


if __name__ == "__main__":
    root = tk.Tk()
    app  = AnimalClassifierApp(root)
    root.mainloop()
