Drone Human & Car Detection System

A computer vision pipeline for detecting and counting humans and cars in drone imagery using PyTorch Faster R-CNN.

Project Overview
It implements a full pipeline from dataset understanding to model training, inference, and evaluation.

Dataset
- Name: VisDrone2019-DET
- Source: [Kaggle - VisDrone Dataset](https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset)
- Train images:6,471
- Val images:548
- Classes detected:Person, Car

Technological Stack
- Python 3.10
- PyTorch + TorchVision
- Faster R-CNN (ResNet50 + FPN backbone)
- OpenCV
- Google Colab (T4 GPU)

How to Run
1. Install dependencies
```bash
pip install torch torchvision opencv-python-headless matplotlib tqdm
```
2. Clone the repo
```bash
git clone https://github.com/TasmiahBinteIqbal/drone-human-detection
cd drone-human-detection
```
3. Run the notebook
Open `drone_detection.ipynb` in Google Colab and run all cells.

Results
Training Loss
Loss decreased from 0.86 → 0.59 over 10 epochs showing consistent learning.

Detection Results
| Metric | Person | Car | Overall |
|--------|--------|-----|---------|
| Precision | 0.423 | 0.410 | 0.417 |
| Recall | 0.627 | 0.529 | 0.578 |
| F1 Score | 0.505 | 0.462 | 0.484 |

Model Details
- Architecture: Faster R-CNN with ResNet50 + FPN backbone
- Pretrained on: COCO dataset
- Fine-tuned on:VisDrone2019-DET-train
- Epochs: 10
- Optimizer: SGD (lr=0.005, momentum=0.9)
- Classes: 2 (Person, Car) + background

Strengths
- Pretrained backbone gives strong feature extraction
- Reasonable recall (62.7%) on challenging drone imagery
- Fast inference (~6.86 it/s on T4 GPU)

Limitations
- Small objects in drone images are hard to detect
- More epochs would improve precision
- Faster R-CNN is slower compared to YOLO-based models

Repository Structure
```
drone-human-detection/
├── drone_detection.ipynb    # Main notebook
├── README.md
├── sample_images.png
├── annotated_samples.png
├── training_loss.png
├── detection_results.png
└── counting_stats.png
```
