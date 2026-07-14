# 🧠 Tài Liệu Kỹ Thuật — Mô Hình Nhận Diện Động Vật

---

## 1. Tổng Quan

Module nhận diện động vật sử dụng kỹ thuật **Transfer Learning** (học chuyển tiếp) với nền tảng là mô hình **MobileNetV3-Large** đã được huấn luyện trước trên bộ dữ liệu **ImageNet** (1.2 triệu ảnh, 1000 lớp).

Thay vì huấn luyện từ đầu — tốn nhiều dữ liệu và thời gian — chúng ta **tận dụng lại** bộ trọng số đã học tốt từ ImageNet, sau đó **fine-tune** (tinh chỉnh) lớp phân loại cuối để nhận diện 10 loài động vật cụ thể của bài toán này.

---

## 2. Tại Sao Chọn MobileNetV3-Large?

| Tiêu chí | MobileNetV3-Large | VGG16 | ResNet50 | EfficientNetB0 |
|----------|:---:|:---:|:---:|:---:|
| Kích thước mô hình | **~21 MB** | ~528 MB | ~98 MB | ~29 MB |
| Tốc độ inference | ⚡ Rất nhanh | 🐢 Chậm | 🚶 Trung bình | ⚡ Nhanh |
| Độ chính xác (ImageNet Top-1) | **75.2%** | 71.5% | 76.1% | 77.1% |
| Phù hợp thiết bị yếu | ✅ Có | ❌ Không | ⚠️ Hạn chế | ✅ Có |
| Hỗ trợ PyTorch | ✅ | ✅ | ✅ | ✅ |

> ✅ **MobileNetV3-Large** là lựa chọn tối ưu: **nhẹ, nhanh, chính xác cao**, phù hợp với cả GPU yếu (RTX 3050 4GB).

---

## 3. Kiến Trúc MobileNetV3-Large

MobileNetV3-Large được xây dựng dựa trên các khối **Inverted Residual** (từ MobileNetV2) kết hợp với cơ chế **Squeeze-and-Excitation (SE)** để học chú ý vào kênh thông tin quan trọng.

```
Input ảnh (224 × 224 × 3)
        │
        ▼
┌─────────────────────────────────────────┐
│  Conv2d (3→16, stride=2) + BN + H-swish │  ← 112×112
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  15 × Bottleneck Block                  │
│  (Depthwise Separable Conv + SE Module) │  ← Backbone
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Conv2d (→ 960) + BN + H-swish          │
│  AdaptiveAvgPool2d (→ 1×1)              │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Classifier (thay thế cho bài toán này) │
│  ┌──────────────────────────────────┐   │
│  │ Linear(960 → 1280) + H-swish    │   │
│  │ Dropout(p=0.2)                   │   │
│  │ Linear(1280 → 10)  ← output     │   │  ← 10 loài động vật
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
        │
        ▼
   Softmax → Xác suất 10 lớp
```

### Các thành phần đặc biệt:

#### 🔹 Depthwise Separable Convolution
Thay vì 1 phép tích chập thông thường, MobileNet tách ra 2 bước:
- **Depthwise Conv**: Lọc từng kênh màu độc lập → giảm ~8× số phép tính
- **Pointwise Conv (1×1)**: Kết hợp thông tin các kênh lại

#### 🔹 Squeeze-and-Excitation (SE)
Cơ chế **chú ý theo kênh (channel attention)**:
1. **Squeeze**: Global Average Pooling → vector đặc trưng toàn cục
2. **Excite**: 2 lớp fully-connected → tạo trọng số cho từng kênh
3. Nhân trọng số vào feature map → mô hình học "kênh nào quan trọng hơn"

#### 🔹 Hard-Swish (H-swish) Activation
Hàm kích hoạt hiệu quả hơn ReLU, được thiết kế để tính toán nhanh trên phần cứng nhúng:
```
H-swish(x) = x · ReLU6(x + 3) / 6
```

---

## 4. Kỹ Thuật Transfer Learning Được Áp Dụng

### 4.1 Fine-Tuning Toàn Mạng (Full Fine-tuning)
Dự án này sử dụng chiến lược **fine-tune toàn bộ mạng**:
- Tải trọng số pre-trained từ ImageNet
- **Không đóng băng (freeze)** bất kỳ lớp nào
- Sử dụng learning rate nhỏ `1e-4` để không làm mất đi kiến thức đã học

```python
# Tất cả tham số đều được cập nhật
optimizer = optim.Adam(model.parameters(), lr=1e-4)
```

### 4.2 Tại sao không đóng băng backbone?
Dataset của chúng ta (4099 ảnh × 10 lớp ≈ ~400 ảnh/lớp) đủ lớn để fine-tune toàn bộ mô hình mà không lo overfitting, đặc biệt khi kết hợp Data Augmentation mạnh.

---

## 5. Tiền Xử Lý Dữ Liệu (Data Pipeline)

### 5.1 Tập Train (Tăng cường dữ liệu)

```
Ảnh gốc
    │
    ├─ Resize(256)               → Đổi kích thước ngắn nhất thành 256px
    ├─ RandomResizedCrop(224)    → Cắt ngẫu nhiên vùng 224×224 (scale 80%–100%)
    ├─ RandomHorizontalFlip()    → Lật ngang ngẫu nhiên (50%)
    ├─ RandomRotation(±15°)      → Xoay ngẫu nhiên ±15 độ
    ├─ ColorJitter(b=0.2,c=0.2,s=0.2) → Thay đổi màu sắc ngẫu nhiên
    ├─ ToTensor()                → Chuyển sang tensor [0,1]
    └─ Normalize(ImageNet mean/std)  → Chuẩn hóa về phân phối ImageNet
         mean = [0.485, 0.456, 0.406]
         std  = [0.229, 0.224, 0.225]
```

### 5.2 Tập Test (Không tăng cường)

```
Ảnh gốc
    │
    ├─ Resize(256)
    ├─ CenterCrop(224)       → Cắt chính giữa (deterministic)
    ├─ ToTensor()
    └─ Normalize(ImageNet mean/std)
```

---

## 6. Cấu Hình Huấn Luyện

| Tham số | Giá trị | Lý do |
|---------|---------|-------|
| Epochs | 10 | Đủ cho transfer learning hội tụ |
| Batch size | 32 | Phù hợp VRAM 4GB |
| Learning rate | `1e-4` | Nhỏ để giữ kiến thức ImageNet |
| Optimizer | **Adam** | Hội tụ nhanh, ít cần điều chỉnh LR |
| Loss function | **CrossEntropyLoss** | Bài toán phân loại nhiều lớp |
| LR Scheduler | **ReduceLROnPlateau** | Giảm LR khi val_acc không tăng (patience=2) |
| Device | **CUDA (RTX 3050)** | Tăng tốc huấn luyện |

---

## 7. Quá Trình Suy Luận (Inference)

Khi nhận một ảnh đầu vào mới:

```
Ảnh đầu vào
    │
    ▼
Tiền xử lý (Resize → CenterCrop → Normalize)
    │
    ▼
Đưa qua MobileNetV3 (forward pass)
    │
    ▼
Vector đặc trưng 10 chiều (logits)
    │
    ▼
Softmax → Vector xác suất [p1, p2, ..., p10]
    │
    ▼
argmax → Lớp có xác suất cao nhất
    │
    ▼
Kết quả: "Tiger" với độ tin cậy 97.5%
```

---

## 8. Đánh Giá Mô Hình

### Chỉ số theo dõi trong quá trình huấn luyện:
- **Train Loss / Test Loss**: Loss giảm theo epoch → mô hình đang học tốt
- **Train Accuracy / Test Accuracy**: Độ chính xác trên tập train và test

### Điều kiện lưu mô hình tốt nhất:
Mô hình chỉ được lưu vào `best_model.pth` khi `Test Accuracy` tại epoch hiện tại **cao hơn** tất cả các epoch trước.

```python
if epoch_val_acc > best_acc:
    best_acc = epoch_val_acc
    torch.save(model.state_dict(), "best_model.pth")
```

---

## 9. Độ Phức Tạp Tính Toán

| Thông số | Giá trị |
|----------|---------|
| Số tham số mô hình | ~5.48 triệu |
| Kích thước file `.pth` | ~21 MB |
| FLOPs mỗi ảnh (224×224) | ~219 MFLOPs |
| Thời gian inference (GPU) | < 10ms |
| Thời gian inference (CPU) | < 100ms |

---

## 10. Thư Viện Sử Dụng

| Thư viện | Phiên bản | Mục đích |
|----------|-----------|----------|
| `torch` | 2.5.1+cu118 | Framework học sâu chính |
| `torchvision` | 0.20.1+cu118 | MobileNetV3, ImageFolder, Transforms |
| `Pillow` | latest | Đọc và xử lý ảnh (thay thế OpenCV) |
| `matplotlib` | 3.10.7 | Vẽ biểu đồ loss/accuracy |
| `tqdm` | latest | Thanh tiến trình huấn luyện |
| `tkinter` | built-in | Giao diện đồ họa (GUI) |

---

## 📚 Tài Liệu Tham Khảo

- **Bài báo gốc MobileNetV3**: *Searching for MobileNetV3* — Howard et al., 2019.  
  [https://arxiv.org/abs/1905.02244](https://arxiv.org/abs/1905.02244)
- **PyTorch Documentation — MobileNetV3**:  
  [https://pytorch.org/vision/stable/models/mobilenetv3.html](https://pytorch.org/vision/stable/models/mobilenetv3.html)
- **Transfer Learning Tutorial (PyTorch)**:  
  [https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
