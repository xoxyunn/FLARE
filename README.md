# FLARE: FLood Assessment and Risk Evaluation

[So Hyun Kang](https://github.com/xoxyunn), [Ga San Jhun](https://github.com/BonusP), [Su Yeon Kim](https://www.linkedin.com/in/%EB%AE%A4-undefined-485772259/), [Min Jun Kim](https://github.com/MinjunKimhongju), Tae Hyung Kim, and Youn Kyu Lee

A region-aware intelligent framework for flood risk estimation that improves robustness to environmental variability. FLARE leverages topographical and environmental features from surveillance cameras to identify regions of interest using deep learning and estimate flood conditions, enabling reliable and region-specific risk assessment across diverse urban environments.

![FLARE architecture](utils/images/main_figure.png?raw=true)


# Flood Estimation from Video

FLARE requires three deep learning models to operate:

- **YOLOv6** — object detection model for identifying vehicles in each frame
- **SAM 2** — image segmentation model for generating water-mask
- **VGG-16** — image classification model for estimating flood condition relative to detected vehicles

## Repository structure

```
.
├── input_video/          # Input images or video files
├── outputs/
│   ├── yolo/car/         # Detected vehicle images and label files
│   ├── images/           # Processed frame images
│   └── result.json       # Final estimation results
└── utils/
    ├── model_weights/    # Pre-trained model weights
    ├── YOLOv6/           # YOLOv6 source (cloned)
    └── sam2/             # SAM 2 source (cloned)
```

## Requirements

- Python 3.10+
- PyTorch = **2.5.1** 


## Installation

### 1. Clone dependency repositories

```bash
cd utils

# YOLOv6
git clone https://github.com/meituan/YOLOv6.git

# Move infer.py one level up
cp ./utils/YOLOv6/tools/infer.py ./utils/YOLOv6/infer.py

# SAM 2
git clone https://github.com/facebookresearch/sam2.git 
```

### 2. Create and activate virtual environment

```bash
python -m venv flare
# Windows (PowerShell)
.\flare\Scripts\Activate.ps1
# Linux / macOS
source flare/bin/activate
```

### 3. Install dependencies

```bash
# Install PyTorch 2.5.1 with CUDA 12.1 (recommended)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# YOLOv6 requirements
pip install -r ./utils/YOLOv6/requirements.txt

# SAM 2
pip install -e ./utils/sam2

# Additional packages
pip install requests psutil matplotlib
```

## Model weights

Download the following weights and place them in `./model_weights/`.

| Model | Source | File |
|---|---|---|
| YOLOv6-L6 | [meituan/YOLOv6](https://github.com/meituan/YOLOv6) | `yolov6l6.pt` |
| SAM 2.1 Large | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | `sam2.1_hiera_large.pt` |
| VGG-16 | [Link](https://drive.google.com/file/d/1ViBMQcCfYUd6za6czs9txHAFzq7mxwfq/view?usp=sharing) | `model_checkpoint_431_0.7600.pt` |

> **SAM 2.1 checkpoint** can also be downloaded directly:
> ```bash
> wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
>     -P ./model_weights/
> ```

## Usage

### Input

Place frame images **or** a video file inside `./input_video/`.

```
input_video/
├── surveillance_frame_001.jpg   # option A — individual frames
└── surveillance_video.mp4       # option B — video file
```

> If using a video file, uncomment and run the frame-extraction cell (cell 2) in the notebook before proceeding.

### Run

Open and run `FLARE.ipynb` top-to-bottom.

### Output

| Path | Contents |
|---|---|
| `./outputs/yolo/car/` | Detected vehicle images and YOLO label files |
| `./outputs/images/` | Processed frame images |
| `./outputs/result.json` | Final flood estimation results |



## License

Please refer to the licenses of each upstream project:
- YOLOv6 — [GPL-3.0](https://github.com/meituan/YOLOv6/blob/main/LICENSE)
- SAM 2 — [Apache-2.0](https://github.com/facebookresearch/sam2/blob/main/LICENSE)