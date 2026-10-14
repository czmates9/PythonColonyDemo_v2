# PythonColonyDemo

Small OpenCV demo for detecting circular colony-like candidates in microscopy images.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

Put `.jpg`, `.jpeg`, or `.png` images into the `images` directory.

```powershell
python main.py --input images
```

Optional thresholds:

```powershell
python main.py --input images --brightness 165 --contrast 15
```

Generated annotated images and `report.json` are written to `output/`.
