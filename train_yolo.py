"""
train_yolo.py
Huấn luyện YOLOv8 ở chế độ Classify trên dataset 10 loài động vật.
Hỗ trợ: YOLOv8n-cls (nano) và YOLOv8s-cls (small).

Cấu trúc dataset yêu cầu (đã có sẵn):
    Dataset/
        train/
            Bear/
            Deer/
            ...
        test/
            Bear/
            Deer/
            ...
"""

import os
import time
import yaml
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────── CẤU HÌNH ───────────────────────────
BASE_DIR   = Path(__file__).parent
DATASET    = BASE_DIR / "Dataset"      # chứa train/ và test/
EPOCHS     = 20                        # số epoch huấn luyện
BATCH      = 16                        # batch size
IMG_SIZE   = 224                       # kích thước ảnh đầu vào
DEVICE     = 0                         # 0 = GPU đầu tiên, "cpu" = CPU

# Danh sách mô hình sẽ huấn luyện và so sánh
MODELS_TO_TRAIN = [
    {"name": "yolov8n-cls", "weights": "yolov8n-cls.pt"},  # Nano  (~6 MB)
    {"name": "yolov8s-cls", "weights": "yolov8s-cls.pt"},  # Small (~22 MB)
]
# ─────────────────────────────────────────────────────────────────







def train_one_model(model_name: str, weights: str):
    print(f"\n{'='*60}")
    print(f"  Bắt đầu huấn luyện: {model_name}")
    print(f"{'='*60}")

    model = YOLO(weights)   # tự tải pretrained weights nếu chưa có

    results = model.train(
        data      = str(DATASET),   # ultralytics classify dùng directory, không dùng yaml
        epochs    = EPOCHS,
        batch     = BATCH,
        imgsz     = IMG_SIZE,
        device    = DEVICE,
        project   = str(BASE_DIR / "runs" / "classify"),
        name      = model_name,
        exist_ok  = True,            # tiếp tục nếu thư mục đã tồn tại
        patience  = 5,               # early stopping
        optimizer = "Adam",
        lr0       = 1e-3,
        verbose   = True,
    )

    save_dir = Path(results.save_dir)
    best_pt  = save_dir / "weights" / "best.pt"
    print(f"\n✅ Huấn luyện xong: {model_name}")
    print(f"   Model tốt nhất: {best_pt}")
    return best_pt


def validate_model(weights_path: Path):
    """Chạy đánh giá trên tập test/ và in kết quả."""
    model_name = weights_path.parent.parent.name
    print(f"\n[Đánh giá] {model_name}  →  tập test")
    model   = YOLO(str(weights_path))
    # ultralytics classify val: dùng thư mục test trực tiếp
    metrics = model.val(
        data   = str(DATASET / "test"),
        imgsz  = IMG_SIZE,
        batch  = BATCH,
        device = DEVICE,
        verbose= False,
    )
    top1 = metrics.top1   # Top-1 Accuracy
    top5 = metrics.top5   # Top-5 Accuracy
    print(f"   Top-1 Accuracy: {top1*100:.2f}%")
    print(f"   Top-5 Accuracy: {top5*100:.2f}%")
    return top1, top5


def benchmark_speed(weights_path: Path, n_runs: int = 100):
    """Đo thời gian inference trung bình (ms) trên 1 ảnh."""
    import torch
    from PIL import Image
    import numpy as np

    model  = YOLO(str(weights_path))
    # tạo ảnh giả
    dummy  = Image.fromarray(np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))

    # warm-up
    for _ in range(5):
        model.predict(source=dummy, imgsz=IMG_SIZE, verbose=False)

    # đo tốc độ
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model.predict(source=dummy, imgsz=IMG_SIZE, verbose=False)
    elapsed = (time.perf_counter() - t0) / n_runs * 1000  # ms

    print(f"   Inference speed ({n_runs} runs): {elapsed:.2f} ms/ảnh")
    return elapsed


def print_summary(results: list):
    print("\n" + "="*65)
    print("  BẢNG TỔNG KẾT SO SÁNH CÁC MÔ HÌNH YOLO")
    print("="*65)
    header = f"{'Mô hình':<22} {'Top-1 Acc':>10} {'Top-5 Acc':>10} {'Speed (ms)':>12}"
    print(header)
    print("-"*65)
    for r in results:
        row = (f"{r['name']:<22} "
               f"{r['top1']*100:>9.2f}% "
               f"{r['top5']*100:>9.2f}% "
               f"{r['speed']:>11.2f}")
        print(row)
    print("="*65)


if __name__ == "__main__":
    if not (DATASET / "train").exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục train tại: {DATASET / 'train'}\n"
            "Hãy chạy split_dataset.py trước."
        )

    summary = []

    for cfg in MODELS_TO_TRAIN:
        best_pt = train_one_model(cfg["name"], cfg["weights"])
        top1, top5 = validate_model(best_pt)
        speed      = benchmark_speed(best_pt)

        summary.append({
            "name":  cfg["name"],
            "top1":  top1,
            "top5":  top5,
            "speed": speed,
            "path":  str(best_pt),
        })

    print_summary(summary)
    print("\n✅ Hoàn tất! Kết quả YOLO lưu tại thư mục: runs/classify/")
