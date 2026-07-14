# 🐾 Nhận Diện 10 Loài Động Vật (Animal Detection)

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Dự án ứng dụng Deep Learning để phân loại và nhận diện 10 loài động vật khác nhau. 
Mô hình sử dụng kiến trúc **MobileNetV3-Large** được fine-tune bằng PyTorch, tối ưu cho tốc độ và độ chính xác, hỗ trợ huấn luyện trên GPU (CUDA) và tích hợp giao diện người dùng (GUI) trực quan.

---

## ✨ Tính Năng Nổi Bật

- **Phân loại 10 loài động vật**: Gấu, Nai, Vịt, Cáo, Vẹt, Thỏ, Gấu mèo, Gấu trúc đỏ, Sóc, Hổ.
- **Mô hình siêu nhẹ & nhanh**: Sử dụng MobileNetV3-Large cho độ chính xác cao nhưng vẫn đảm bảo tốc độ suy luận nhanh (phù hợp chạy trên CPU lẫn GPU).
- **Giao diện trực quan (GUI)**: Ứng dụng Desktop cho phép người dùng chọn ảnh và nhận kết quả dự đoán (kèm độ tin cậy) tức thì.
- **Tự động hóa**: Cung cấp các script chia dữ liệu chuẩn xác, huấn luyện mô hình (tự động lưu best model) và đánh giá độ chính xác (Vẽ biểu đồ Loss/Accuracy).

---

## 🛠️ Công Nghệ Sử Dụng

- **Ngôn ngữ**: Python 3.9+
- **Deep Learning Framework**: PyTorch, Torchvision
- **Xử lý ảnh**: Pillow (PIL)
- **Trực quan hóa dữ liệu**: Matplotlib
- **Khác**: Tqdm (hiển thị tiến trình)

> **Lưu ý**: Dự án không sử dụng OpenCV để tránh xung đột với NumPy 2.x, đảm bảo sự ổn định khi triển khai.

---

## 📁 Cấu Trúc Dự Án

```text
vpt-animal-detection/
│
├── Dataset/                   ← Chứa dữ liệu (được sinh ra qua script chia dữ liệu)
│   ├── train/                 ← 80% dữ liệu huấn luyện
│   └── test/                  ← 20% dữ liệu kiểm thử
│
├── app_gui.py                 ← Ứng dụng Desktop (Giao diện người dùng)
├── compare_models.py          ← Script so sánh các mô hình (nếu có)
├── model.py                   ← Định nghĩa kiến trúc MobileNetV3-Large
├── predict.py                 ← Script suy luận qua Command Line
├── split_dataset.py           ← Script tự động chia tập Train/Test
├── train.py                   ← Script huấn luyện mô hình (Train & Eval)
│
├── best_model.pth             ← Trọng số của mô hình tốt nhất (Tạo ra sau khi Train)
├── classes.txt                ← Danh sách 10 nhãn phân loại
├── training_curves.png        ← Biểu đồ quá trình học (Tạo ra sau khi Train)
└── README.md                  ← Tài liệu dự án
```

---

## 🚀 Hướng Dẫn Cài Đặt & Sử Dụng

### 1. Yêu cầu hệ thống
- Python 3.9 trở lên.
- Cài đặt các thư viện cần thiết:
```bash
pip install torch torchvision pillow matplotlib tqdm
```

### 2. Chuẩn bị dữ liệu
Nếu bạn có tập dữ liệu gốc, hãy chạy script sau để tự động chia dữ liệu thành 2 tập `train` (80%) và `test` (20%):
```bash
python split_dataset.py
```

### 3. Huấn luyện mô hình (Training)
Quá trình huấn luyện tự động tải pre-trained weights và fine-tune.
```bash
python train.py
```
> **Kết quả đầu ra**: Mô hình tốt nhất sẽ được lưu tại `best_model.pth`. Biểu đồ huấn luyện lưu tại `training_curves.png`.

### 4. Sử dụng mô hình (Inference)
**Cách 1: Qua dòng lệnh (CLI)**
```bash
python predict.py "đường_dẫn_đến_ảnh.jpg"
```

**Cách 2: Qua giao diện đồ họa (GUI)**
```bash
python app_gui.py
```
Nhấn **"Chọn hình ảnh từ máy tính"**, hình ảnh và kết quả dự đoán sẽ hiện ra trực quan trên màn hình.

---

## 🎥 Video Demo & Kết Quả Đạt Được

**Video Demo quá trình nhận diện:**

https://github.com/cuongherok4/vpt-animal-detection/raw/main/demo_mibilev3.mp4

*(Lưu ý: Github sẽ tự động hiển thị video player cho link trên, nếu không bạn có thể click vào để xem)*

Mô hình đạt độ chính xác cao trên tập dữ liệu kiểm thử. Với cấu trúc MobileNetV3-Large kết hợp các kỹ thuật Transfer Learning và Data Augmentation (Random Crop, Flip, ColorJitter), mô hình có khả năng chống overfitting tốt.

---

## 🤝 Đóng Góp
Mọi đóng góp, báo lỗi (issues) hoặc pull requests đều được chào đón!

## 📝 License
Dự án được phân phối dưới giấy phép MIT License.
