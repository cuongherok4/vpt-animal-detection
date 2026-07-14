# 🐾 Module Nhận Diện Động Vật - Hướng Dẫn Sử Dụng

Module nhận diện **10 loài động vật** sử dụng mô hình học sâu **MobileNetV3-Large** (PyTorch) với GPU CUDA.

---

## 📁 Cấu Trúc Thư Mục

```
Module_dongvat/
│
├── Dataset/
│   ├── train/         ← Dữ liệu huấn luyện (80%)
│   │   ├── Bear/
│   │   ├── Deer/
│   │   └── ... (10 loài)
│   └── test/          ← Dữ liệu kiểm thử (20%)
│       ├── Bear/
│       └── ... (10 loài)
│
├── model.py           ← Định nghĩa kiến trúc mô hình
├── train.py           ← Script huấn luyện mô hình
├── predict.py         ← Dự đoán qua dòng lệnh
├── app_gui.py         ← Giao diện đồ họa (GUI)
├── split_dataset.py   ← Script chia dữ liệu train/test
│
├── best_model.pth     ← (Tự tạo sau khi train) Trọng số mô hình tốt nhất
├── classes.txt        ← (Tự tạo sau khi train) Danh sách tên các lớp
└── training_curves.png← (Tự tạo sau khi train) Biểu đồ quá trình học
```

---

## 🐍 Yêu Cầu Hệ Thống

| Thành phần | Phiên bản |
|---|---|
| Python | 3.9 – 3.11 |
| PyTorch | 2.5+ (CUDA 11.8) |
| Torchvision | 0.20+ |
| Pillow | Bất kỳ |
| Matplotlib | Bất kỳ |
| tqdm | Bất kỳ |

> **Lưu ý:** Dự án đã được kiểm thử với GPU **NVIDIA RTX 3050 (CUDA)**. Mô hình vẫn có thể chạy trên CPU nhưng sẽ chậm hơn.

---

## ⚠️ Vấn đề Thường Gặp: Xung đột NumPy + OpenCV

Hệ thống đang dùng NumPy 2.x, nên **OpenCV (cv2) không tương thích**. Module này đã được viết hoàn toàn **không phụ thuộc vào OpenCV**, chỉ dùng **Pillow (PIL)** để xử lý ảnh.

---

## 🚀 Hướng Dẫn Từng Bước

### Bước 1 — Chia dữ liệu Train/Test (đã thực hiện)

> Nếu thư mục `Dataset/train/` và `Dataset/test/` đã tồn tại, hãy **bỏ qua bước này**.

```powershell
cd "d:\TÀI LIỆU NĂM 4\Module_dongvat"
python split_dataset.py
```

✅ Kết quả: Chia 4.099 ảnh thành 3.275 ảnh train và 824 ảnh test.

---

### Bước 2 — Huấn Luyện Mô Hình

```powershell
cd "d:\TÀI LIỆU NĂM 4\Module_dongvat"
python train.py
```

**Quá trình huấn luyện sẽ:**
- Tải trọng số pre-trained MobileNetV3 từ PyTorch (khoảng ~21MB, chỉ tải 1 lần)
- Huấn luyện qua **10 epochs** trên GPU
- In tiến trình mỗi batch (train loss, accuracy)
- **Tự động lưu mô hình tốt nhất** vào `best_model.pth`
- Lưu danh sách lớp vào `classes.txt`
- Lưu biểu đồ Loss & Accuracy vào `training_curves.png`

**Thời gian ước tính:** ~3–10 phút tùy vào GPU.

**Ví dụ kết quả kỳ vọng:**
```
Epoch 10 Results:
  Train Loss: 0.2431 | Train Acc: 92.50%
  Test Loss:  0.3105 | Test Acc:  89.20%
=> Saved new best model with test accuracy: 89.20%
```

---

### Bước 3 — Dự Đoán Qua Dòng Lệnh (Tùy Chọn)

Sau khi huấn luyện xong, bạn có thể kiểm tra nhanh 1 ảnh bất kỳ:

```powershell
cd "d:\TÀI LIỆU NĂM 4\Module_dongvat"
python predict.py "đường_dẫn_đến_ảnh.jpg"
```

**Ví dụ:**
```powershell
python predict.py "C:\Users\cuong\Pictures\tiger.jpg"
```

**Kết quả:**
```
Predicted: Tiger
Confidence: 97.85%
```

---

### Bước 4 — Chạy Giao Diện Đồ Họa (GUI)

```powershell
cd "d:\TÀI LIỆU NĂM 4\Module_dongvat"
python app_gui.py
```

**Tính năng GUI:**
- 🟢 Nút **"Chọn hình ảnh từ máy tính"** — Mở hộp thoại chọn ảnh (`.jpg`, `.png`, `.webp`, ...)
- Hiển thị ảnh preview và **tên loài + độ tin cậy %** ngay sau khi chọn
- 🟣 Nút **"Nạp lại mô hình"** — Tải lại model nếu vừa cập nhật `best_model.pth`

---

## 🐾 10 Loài Động Vật Được Nhận Diện

| STT | Tên loài | Số ảnh train | Số ảnh test |
|-----|----------|-------------|-------------|
| 1 | Bear (Gấu) | 266 | 67 |
| 2 | Deer (Nai) | 416 | 105 |
| 3 | Duck (Vịt) | 467 | 117 |
| 4 | Fox (Cáo) | 339 | 85 |
| 5 | Parrot (Vẹt) | 436 | 110 |
| 6 | Rabbit (Thỏ) | 252 | 64 |
| 7 | Raccoon (Gấu mèo) | 256 | 64 |
| 8 | Red panda (Gấu trúc đỏ) | 260 | 66 |
| 9 | Squirrel (Sóc) | 343 | 86 |
| 10 | Tiger (Hổ) | 240 | 60 |

---

## 🛠️ Kiến Trúc Mô Hình

```
MobileNetV3-Large (pretrained trên ImageNet)
    └── Backbone: Feature Extraction layers
    └── Classifier:
        ├── Linear(960 → 1280) + Hardswish
        ├── Dropout(0.2)
        └── Linear(1280 → 10)  ← Fine-tuned cho 10 loài
```

**Kỹ thuật huấn luyện:**
- **Transfer Learning**: Dùng trọng số ImageNet, fine-tune toàn bộ mạng
- **Data Augmentation**: Random Crop, Flip, Rotation, ColorJitter
- **Optimizer**: Adam (lr=1e-4)
- **Scheduler**: ReduceLROnPlateau (giảm LR khi val_acc không tăng)

---

## 📊 Kiểm Tra Biểu Đồ Huấn Luyện

Sau khi train xong, mở file `training_curves.png` để xem đồ thị:
- **Trái**: Đường Loss của Train và Test theo từng epoch
- **Phải**: Đường Accuracy của Train và Test theo từng epoch

---

## 📌 Ghi Chú Quan Trọng

> [!NOTE]
> File `best_model.pth` và `classes.txt` phải nằm **cùng thư mục** với `app_gui.py` và `predict.py`.

> [!WARNING]
> **Không xóa** `classes.txt`! File này quy định thứ tự nhãn tương ứng với đầu ra của mô hình. Nếu xóa, mô hình sẽ dự đoán sai tên loài.

> [!TIP]
> Để tăng độ chính xác, bạn có thể tăng số epoch trong `train.py` tại dòng:
> ```python
> train_model(dataset_dir, epochs=20, batch_size=32, lr=1e-4)
> ```
