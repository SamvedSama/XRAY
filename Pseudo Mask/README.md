# DDRU-Net for Lung Segmentation on Chest X-rays

This project implements a Dense Dilated Residual U-Net (DDRU-Net) for lung segmentation on chest X-ray images.

## Features

- DenseNet201 encoder backbone
- Dilated bottleneck convolutions
- Residual decoder blocks
- Dice Loss
- IoU, Dice, Hausdorff metrics
- GPU-ready TensorFlow/Keras implementation
- Save/load reusable model and weights
- Save train/val/test predictions with original image names

---

## Folder Structure

```text
ddru-net-lung-segmentation/
│
├── train.py
├── inference.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── MC/
│   │   ├── images/
│   │   └── masks/
│   └── SH/
│       ├── images/
│       └── masks/
│
└── outputs/
    ├── checkpoints/
    ├── weights/
    ├── logs/
    ├── splits/
    ├── predictions/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── plots/
```
