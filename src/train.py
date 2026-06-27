from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path(__file__).resolve().parent.parent / "data" / "data.yaml"


def main():
    model = YOLO("yolo11n.pt")
    results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,
        project="runs",
    )

    print(f"Final metrics: {results.save_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
