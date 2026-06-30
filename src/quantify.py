import argparse
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

WEIGHTS = Path(__file__).resolve().parent.parent / "runs" / "detect" / "runs" / "train-3" / "weights" / "best.pt"


def main():
    parser = argparse.ArgumentParser(description="Quantify microplastic particles in a folder of images")
    parser.add_argument("images_dir", type=Path, help="Folder of microscopy images to run inference on")
    args = parser.parse_args()

    model = YOLO(str(WEIGHTS))
    results = model.predict(source=str(args.images_dir), stream=True)

    for result in results:
        boxes = result.boxes
        class_counts = Counter(model.names[int(cls_id)] for cls_id in boxes.cls)

        print(f"\n{Path(result.path).name}")
        print(f"  Total particles: {len(boxes)}")
        print(f"  Per-class count: {dict(class_counts)}")

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width_px = x2 - x1
            height_px = y2 - y1
            class_name = model.names[int(box.cls[0])]
            print(f"    {class_name}: {width_px:.1f}px x {height_px:.1f}px")


if __name__ == "__main__":
    main()
