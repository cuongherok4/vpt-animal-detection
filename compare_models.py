"""
compare_models.py
So sánh hiệu suất giữa các mô hình YOLO và MobileNetV3.

Xuất ra:
  - Bảng so sánh trên terminal
  - Biểu đồ cột so sánh: comparison_chart.png

Lưu ý: Cần chạy train_yolo.py trước để có file dataset.yaml và best.pt.
"""

import os
import time
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR   = Path(__file__).parent
DATASET    = BASE_DIR / "Dataset"
YAML_PATH  = BASE_DIR / "dataset.yaml"   # được tạo bởi train_yolo.py
IMG_SIZE   = 224
N_SPEED    = 100   # số lần đo inference speed

# ────────── Cấu hình các mô hình cần so sánh ──────────
YOLO_MODELS = [
    {
        "name":    "YOLOv8n-cls\n(Nano, ~6 MB)",
        "label":   "YOLOv8n-cls",
        "path":    BASE_DIR / "runs" / "classify" / "yolov8n-cls" / "weights" / "best.pt",
        "type":    "yolo",
        "color":   "#FF6B6B",
    },
    {
        "name":    "YOLOv8s-cls\n(Small, ~22 MB)",
        "label":   "YOLOv8s-cls",
        "path":    BASE_DIR / "runs" / "classify" / "yolov8s-cls" / "weights" / "best.pt",
        "type":    "yolo",
        "color":   "#FF9F43",
    },
]

MOBILENET_MODEL = {
    "name":    "MobileNetV3-Large\n(~21 MB)",
    "label":   "MobileNetV3",
    "path":    BASE_DIR / "best_model.pth",
    "classes": BASE_DIR / "classes.txt",
    "type":    "mobilenet",
    "color":   "#54A0FF",
}
# ───────────────────────────────────────────────────────


def load_classes(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


# ─── Đánh giá YOLO ───
def evaluate_yolo(cfg: dict):
    from ultralytics import YOLO

    if not cfg["path"].exists():
        print(f"  ⚠️  Chưa tìm thấy: {cfg['path']}")
        print(f"  → Hãy chạy train_yolo.py trước.")
        return None

    model = YOLO(str(cfg["path"]))

    # Accuracy trên tập test (dùng yaml, split val → test/)
    if not YAML_PATH.exists():
        print(f"  ⚠️  Chưa có dataset.yaml — hãy chạy train_yolo.py trước.")
        return None

    metrics = model.val(
        data=str(YAML_PATH), split="val",
        imgsz=IMG_SIZE, batch=16, verbose=False
    )
    top1 = metrics.top1
    top5 = metrics.top5

    # Speed
    dummy = Image.fromarray(
        np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    )
    for _ in range(10):
        model.predict(source=dummy, imgsz=IMG_SIZE, verbose=False)
    t0 = time.perf_counter()
    for _ in range(N_SPEED):
        model.predict(source=dummy, imgsz=IMG_SIZE, verbose=False)
    speed = (time.perf_counter() - t0) / N_SPEED * 1000

    # Kích thước file (MB)
    size_mb = cfg["path"].stat().st_size / 1e6

    return {"top1": top1, "top5": top5, "speed": speed, "size_mb": size_mb}


# ─── Đánh giá MobileNetV3 ───
def evaluate_mobilenet(cfg: dict):
    from torchvision import datasets, transforms, models
    import torch.nn as nn

    if not cfg["path"].exists():
        print(f"  ⚠️  Chưa tìm thấy: {cfg['path']}")
        print(f"  → Hãy chạy train.py trước.")
        return None

    classes = load_classes(cfg["classes"])
    num_cls = len(classes)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load mô hình
    model = models.mobilenet_v3_large()
    in_f  = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_f, num_cls)
    model.load_state_dict(torch.load(cfg["path"], map_location=device))
    model = model.to(device).eval()

    # DataLoader tập test
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_ds = datasets.ImageFolder(str(DATASET / "test"), transform=tfm)
    loader  = torch.utils.data.DataLoader(test_ds, batch_size=32,
                                          shuffle=False, num_workers=2)

    correct1 = correct5 = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)

            # Top-1
            _, pred1 = out.max(1)
            correct1 += pred1.eq(labels).sum().item()

            # Top-5
            _, pred5 = out.topk(min(5, num_cls), 1, True, True)
            pred5    = pred5.t()
            correct5 += pred5.eq(labels.view(1, -1).expand_as(pred5)).any(0).sum().item()

            total += labels.size(0)

    top1 = correct1 / total
    top5 = correct5 / total

    # Speed
    dummy_t = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    for _ in range(10):
        model(dummy_t)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N_SPEED):
            model(dummy_t)
    speed = (time.perf_counter() - t0) / N_SPEED * 1000

    size_mb = cfg["path"].stat().st_size / 1e6

    return {"top1": top1, "top5": top5, "speed": speed, "size_mb": size_mb}


# ─── Vẽ biểu đồ so sánh ───
def plot_comparison(names, labels, top1_list, speed_list, colors):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#1E1E2E")
    for ax in (ax1, ax2):
        ax.set_facecolor("#282A36")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#6272A4")

    x = range(len(labels))

    # Biểu đồ 1: Top-1 Accuracy
    bars1 = ax1.bar(x, [v * 100 for v in top1_list], color=colors,
                    width=0.5, edgecolor="#44475A", linewidth=1.2)
    ax1.set_title("Top-1 Accuracy (%)\n(cao hơn = tốt hơn)",
                  color="white", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, color="white", fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Accuracy (%)", color="white", fontsize=11)
    for bar, val in zip(bars1, top1_list):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1.5,
                 f"{val*100:.1f}%",
                 ha="center", va="bottom", color="white", fontsize=11,
                 fontweight="bold")

    # Biểu đồ 2: Inference Speed
    bars2 = ax2.bar(x, speed_list, color=colors,
                    width=0.5, edgecolor="#44475A", linewidth=1.2)
    ax2.set_title("Tốc độ Inference (ms/ảnh)\n(thấp hơn = nhanh hơn)",
                  color="white", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, color="white", fontsize=10)
    ax2.set_ylabel("Thời gian (ms)", color="white", fontsize=11)
    for bar, val in zip(bars2, speed_list):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.2,
                 f"{val:.1f} ms",
                 ha="center", va="bottom", color="white", fontsize=11,
                 fontweight="bold")

    plt.suptitle("So Sánh Hiệu Suất Các Mô Hình Nhận Diện Động Vật",
                 color="white", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = BASE_DIR / "comparison_chart.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n📊 Đã lưu biểu đồ: {out}")
    plt.show()


# ─── Bảng kết quả terminal ───
def print_table(rows):
    print("\n" + "=" * 72)
    print("  BÁO CÁO SO SÁNH MÔ HÌNH — NHẬN DIỆN ĐỘNG VẬT (10 LOÀI)")
    print("=" * 72)
    fmt = "{:<24} {:>10} {:>10} {:>12} {:>10}"
    print(fmt.format("Mô hình", "Top-1 (%)", "Top-5 (%)", "Speed (ms)", "Size (MB)"))
    print("-" * 72)
    for r in rows:
        print(fmt.format(
            r["label"],
            f"{r['top1']*100:.2f}",
            f"{r['top5']*100:.2f}",
            f"{r['speed']:.2f}",
            f"{r['size_mb']:.1f}",
        ))
    print("=" * 72)

    # Tìm model tốt nhất
    best_acc   = max(rows, key=lambda r: r["top1"])
    best_speed = min(rows, key=lambda r: r["speed"])
    print(f"\n🏆 Accuracy cao nhất : {best_acc['label']} — {best_acc['top1']*100:.2f}%")
    print(f"⚡ Tốc độ nhanh nhất : {best_speed['label']} — {best_speed['speed']:.2f} ms/ảnh")


if __name__ == "__main__":
    all_cfgs = YOLO_MODELS + [MOBILENET_MODEL]
    rows     = []

    for cfg in all_cfgs:
        print(f"\n📦 Đang đánh giá: {cfg['label']} ...")
        if cfg["type"] == "yolo":
            res = evaluate_yolo(cfg)
        else:
            res = evaluate_mobilenet(cfg)

        if res:
            rows.append({**cfg, **res})
        else:
            print(f"  ⏩ Bỏ qua {cfg['label']} (chưa có model).")

    if rows:
        print_table(rows)
        plot_comparison(
            names  = [r["name"]  for r in rows],
            labels = [r["label"] for r in rows],
            top1_list = [r["top1"]  for r in rows],
            speed_list = [r["speed"] for r in rows],
            colors = [r["color"] for r in rows],
        )
    else:
        print("\n⚠️  Không có mô hình nào để so sánh. Hãy train trước.")
