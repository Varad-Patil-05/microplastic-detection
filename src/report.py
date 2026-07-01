import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from ultralytics import YOLO

WEIGHTS = Path(__file__).resolve().parent.parent / "runs" / "detect" / "runs" / "train-3" / "weights" / "best.pt"
REPORTS = Path(__file__).resolve().parent.parent / "reports"


def run(images_dir: Path):
    REPORTS.mkdir(exist_ok=True)

    model = YOLO(str(WEIGHTS))
    class_names = list(model.names.values())

    rows = []
    for result in model.predict(source=str(images_dir), stream=True):
        image_name = Path(result.path).name
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width_px = x2 - x1
            height_px = y2 - y1
            rows.append({
                "image_name": image_name,
                "class": model.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 4),
                "width_px": round(width_px, 2),
                "height_px": round(height_px, 2),
                "area_px": round(width_px * height_px, 2),
            })

    if not rows:
        print("No detections found.")
        return

    # CSV
    csv_path = REPORTS / "particles.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "class", "confidence", "width_px", "height_px", "area_px"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {csv_path}  ({len(rows)} detections)")

    # Bar chart — total count per class
    class_counts = {name: sum(1 for r in rows if r["class"] == name) for name in class_names}
    bar_path = REPORTS / "class_counts.png"
    fig, ax = plt.subplots()
    ax.bar(class_counts.keys(), class_counts.values(), color=["#4c72b0", "#dd8452", "#55a868"])
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Detected particle count per class")
    for i, (cls, cnt) in enumerate(class_counts.items()):
        ax.text(i, cnt + 0.3, str(cnt), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"Bar chart saved: {bar_path}")

    # Histogram — particle size (area_px)
    areas = [r["area_px"] for r in rows]
    hist_path = REPORTS / "size_histogram.png"
    fig, ax = plt.subplots()
    ax.hist(areas, bins=30, color="#4c72b0", edgecolor="white")
    ax.set_xlabel("Area (px²)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of particle sizes (area in pixels²)")
    fig.tight_layout()
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"Histogram saved: {hist_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate detection report from a folder of microscopy images")
    parser.add_argument("images_dir", type=Path, help="Folder of images to run inference on")
    args = parser.parse_args()
    run(args.images_dir)


if __name__ == "__main__":
    main()
