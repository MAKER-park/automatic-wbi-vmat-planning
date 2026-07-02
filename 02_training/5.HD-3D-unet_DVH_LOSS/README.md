# Medical Dose Parameter Optimization & Prediction

This project focuses on optimizing radiation therapy dose distributions using deep learning.

## New Feature: True 3D HD-Unet Training

We have added a true 3D training script: `0.train-HD-3D-Unet-DVH_loss.py`.

### Key Improvements
- **Architecture:** Implements a high-depth 3D U-Net (`HD_UNet`) with 5 levels of encoding/decoding, using 3D convolutions and trilinear interpolation.
- **Input Strategy:** Transitioned from 2.5D slice-stacking to true 3D patch-based training.
  - Patch Size: `(32, 256, 256)` (Depth, Height, Width).
  - This satisfies the U-Net's requirement for dimensions divisible by 32.
- **3D DVH Loss:** The Dose-Volume Histogram (DVH) constraint loss has been adapted to compute penalties over 3D volumes rather than 2D slices.
- **Memory Efficiency:** Uses a small batch size and 3D patches to keep VRAM usage manageable while processing volumetric data.
- **Self-Contained:** Includes its own visualization and DVH plotting logic to avoid dependencies on older 2.5D utilities.

### Usage
To start training the 3D model:
```bash
python 0.train-HD-3D-Unet-DVH_loss.py
```

### Data Requirements
- CT and Contour data: `.npy` files in `../final_dataset/Train/CT_and_Contour`.
- Dose data: `.npy` files in `../final_dataset/Train/Dose`.
- Shape: `(H, W, Z, C)` for input and `(H, W, Z)` for dose.
