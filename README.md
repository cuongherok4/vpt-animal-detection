# 🐾 Nhận Diện 10 Loài Động Vật (Animal Detection)

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Dự án ứng dụng Deep Learning để phân loại và nhận diện 10 loài động vật khác nhau. 
Hệ thống hiện tại hỗ trợ và so sánh hiệu năng giữa các kiến trúc mạng: **MobileNetV3-Large**, **YOLOv8n-cls** (Nano), và **YOLOv8s-cls** (Small).

---

## ✨ Tính Năng Nổi Bật

- **Phân loại 10 loài động vật**: Gấu, Nai, Vịt, Cáo, Vẹt, Thỏ, Gấu mèo, Gấu trúc đỏ, Sóc, Hổ.
- **Hỗ trợ đa mô hình**: 
  - **MobileNetV3-Large**: Fine-tune bằng PyTorch thuần, nhẹ và tối ưu CPU/GPU.
  - **YOLOv8-cls**: Tích hợp bộ thư viện siêu mạnh của Ultralytics (bản Nano và Small) cho bài toán classification.
- **So sánh & Đánh giá (Benchmark)**: Cung cấp script tự động huấn luyện, validate, tính toán Top-1/Top-5 Accuracy và đo tốc độ inference (ms) để chọn ra mô hình tốt nhất.
- **Giao diện trực quan (GUI)**: Ứng dụng Desktop cho phép chọn ảnh và nhận kết quả dự đoán kèm độ tin cậy tức thì.
- **Tự động hóa dữ liệu**: Script tự động chia tập dữ liệu `train/test` (tỉ lệ 80/20).

---

## 🛠️ Công Nghệ Sử Dụng

- **Ngôn ngữ**: Python 3.9+
- **Deep Learning Framework**: PyTorch, Torchvision, Ultralytics (YOLOv8)
- **Xử lý ảnh**: Pillow (PIL)
- **Khác**: Tqdm, Matplotlib

> **Lưu ý**: GUI của dự án sử dụng Pillow để xử lý ảnh, tránh xung đột OpenCV khi triển khai. YOLO được dùng riêng cho việc so sánh và đánh giá cấu trúc.

---

## 📁 Cấu Trúc Dự Án

```text
vpt-animal-detection/
│
├── Dataset/                   ← Dữ liệu (sinh ra qua script chia tập)
├── runs/                      ← Thư mục lưu kết quả training YOLO tự động
│
├── app_gui.py                 ← Ứng dụng Desktop (Giao diện người dùng)
├── model.py                   ← Định nghĩa kiến trúc MobileNetV3-Large
├── predict.py                 ← Script suy luận cho MobileNetV3 qua CLI
├── split_dataset.py           ← Script chia tập Train/Test (80/20)
├── train.py                   ← Huấn luyện MobileNetV3
├── train_yolo.py              ← Huấn luyện & benchmark YOLOv8n-cls và YOLOv8s-cls
├── compare_models.py          ← Script so sánh các mô hình
│
├── best_model.pth             ← Trọng số MobileNetV3 tốt nhất sau khi train
├── yolov8n-cls.pt             ← Trọng số pretrained YOLOv8 Nano
├── yolov8s-cls.pt             ← Trọng số pretrained YOLOv8 Small
├── classes.txt                ← Danh sách 10 nhãn phân loại
└── README.md                  ← Tài liệu dự án
```

---

## 🚀 Hướng Dẫn Cài Đặt & Sử Dụng

### 1. Yêu cầu hệ thống
- Python 3.9 trở lên.
- Cài đặt thư viện:
```bash
pip install torch torchvision ultralytics pillow matplotlib tqdm
```

### 2. Chuẩn bị dữ liệu
Chạy script sau để tự động chia dữ liệu thành 2 tập `train` (80%) và `test` (20%):
```bash
python split_dataset.py
```

### 3. Huấn luyện và So sánh (Training & Benchmark)

**Huấn luyện MobileNetV3:**
```bash
python train.py
```

**Huấn luyện YOLOv8 (Nano & Small) và xem bảng so sánh Benchmark:**
```bash
python train_yolo.py
```

### 4. Sử dụng mô hình (Inference)
Chạy ứng dụng giao diện (GUI) để kiểm thử mô hình bằng trực quan:
```bash
python app_gui.py
```

---

## 🎥 Video Demo & Kết Quả Đạt Được

[![Video Demo](https://img.youtube.com/vi/WUf8zkJZ4io/0.jpg)](https://www.youtube.com/watch?v=WUf8zkJZ4io)

*(Nhấn vào hình ảnh trên để xem video demo trực tiếp trên YouTube)*

Mô hình đạt độ chính xác cao trên tập dữ liệu kiểm thử. Với sự so sánh giữa **MobileNetV3** và **YOLOv8-cls**, hệ thống đưa ra cái nhìn tổng quan nhất về sự đánh đổi giữa *tốc độ suy luận (Inference Speed)* và *độ chính xác (Top-1/Top-5 Acc)*.

---

## 🤝 Đóng Góp
Mọi đóng góp, báo lỗi (issues) hoặc pull requests đều được chào đón!

## 📝 License
Dự án được phân phối dưới giấy phép MIT License.
