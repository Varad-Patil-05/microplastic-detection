# Microplastic Detection

A YOLOv11 system that detects and quantifies microplastics (bead, fiber, fragment) in microscopy images.

---

## Results

Trained on the [microplastic-v2-wowak](https://universe.roboflow.com/research-new-things-m0fiq/microplastic-v2-wowak) Roboflow dataset (CC BY 4.0), 1,461 train / 140 val images, 50 epochs, YOLOv11n backbone.

| Metric | Value |
|---|---|
| mAP50 | 0.925 |
| mAP50-95 | 0.757 |
| Precision | 0.917 |
| Recall | 0.822 |

**Per-class (validation set):**

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| bead | 0.897 | 0.671 | 0.879 | 0.671 |
| fiber | 0.940 | 0.864 | 0.933 | 0.795 |
| fragment | 0.915 | 0.931 | 0.964 | 0.805 |

Bead recall is the weakest class (0.671) — the model misses more bead instances than fiber or fragment.

---

## Limitations

- **Augmented duplicates in dataset.** The Roboflow dataset contains augmented copies of source images in the training and validation splits. This means metrics may be slightly inflated compared to performance on entirely unseen images.
- **Pixel-only size measurements.** Particle dimensions (width, height, area) are reported in pixels only. No µm calibration is available, so sizes cannot be converted to physical units without a known scale bar per image. Observed sizes skew small, consistent with genuinely microscopic particles. Low-confidence detections were rare across the validation set.
- **Small model.** YOLOv11n was chosen for the 4 GB GPU constraint. A larger variant (YOLOv11s/m) may improve bead recall, though this hasn't been tested.
- **Three-class scope.** Only bead, fiber, and fragment are detected. Other microplastic morphologies (film, foam, pellet) are outside the current training distribution.

---

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env` and fill in your Roboflow API key:

```
ROBOFLOW_API_KEY=your_key_here
```

---

## Usage

**1. Download dataset**
```bash
python src/download_data.py
```
Downloads the microplastic-v2-wowak dataset into `data/` in YOLOv11 format.

**2. Train**
```bash
python src/train.py
```
Trains YOLOv11n for 50 epochs on GPU (device 0), batch 4, imgsz 640. Saves weights to `runs/detect/train/weights/`.

**3. Quantify (per-image summary)**
```bash
python src/quantify.py <images_dir>
```
Prints total particle count, per-class count, and each particle's bounding box size (width × height in pixels) for every image in the folder.

**4. Report (CSV + charts)**
```bash
python src/report.py <images_dir>
```
Saves to `reports/`:
- `particles.csv` — one row per detection: `image_name, class, confidence, width_px, height_px, area_px`
- `class_counts.png` — bar chart of total detections per class
- `size_histogram.png` — histogram of particle area in pixels²

---

## Tech stack

| Component | Library / Tool |
|---|---|
| Detection model | [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) |
| Dataset | Roboflow (`roboflow` SDK) |
| Image processing | OpenCV (`opencv-python`) |
| Data analysis | pandas |
| Charts | matplotlib |
| Config | python-dotenv, PyYAML |
| Training hardware | NVIDIA GeForce RTX 2050 (4 GB) |

---

## Project structure

```
configs/        # Dataset and training config stubs (data.yaml, train.yaml)
data/           # Downloaded dataset — gitignored
notebooks/      # Exploratory analysis
reports/        # Generated CSV and charts — gitignored
runs/           # Training runs and weights — gitignored
src/
  download_data.py   # Roboflow dataset download
  train.py           # YOLOv11n training
  quantify.py        # Per-image particle counts and sizes
  report.py          # CSV export and charts
```
