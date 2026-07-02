# Code guide

## 1. Preprocessing

The CT-first notebook is the primary pipeline. It reads CT, RTSTRUCT and RTDOSE, resamples them onto a common grid, and writes paired arrays. The dose-first notebook is retained as an alternative/reference implementation.

Expected model input is `(H, W, Z, 9)`:

| Channel | Content |
|---:|---|
| 0 | CT in HU |
| 1–8 | Binary ROI masks used during training |

The exact channel-to-ROI order is a model contract. Confirm it in the notebook before preparing a new cohort; silently changing it invalidates the checkpoint.

## 2. Training and dose inference

Four experiments are preserved:

| Directory | Architecture | Context | DVH-aware loss |
|---|---|---|---|
| `1.2D` | 2D U-Net + SimAM | one axial slice | no |
| `2.2.5D-no-dvh` | 2.5D U-Net + SimAM | adjacent slices | no |
| `2.2.5D-with-dvh` | 2.5D U-Net + SimAM | adjacent slices | yes |
| `5.HD-3D-unet_DVH_LOSS` | patch-based HD 3D U-Net | 3D patch `(32,256,256)` | yes; two-stage option |

Training scripts normalize dose around the 26 Gy prescription used by this project. Do not apply the weights to a different prescription, fractionation, anatomy, ROI convention or image geometry without retraining or explicit validation.

The 3D two-stage path trains stage 1, initializes stage 2 from stage 1, performs sliding-patch full-volume inference, and saves `(H,W,Z)` predicted-dose arrays.

## 3. Postprocessing and mimicking-parameter inference

Evaluation scripts calculate dose statistics, ROI DVH endpoints, MAE/SSIM, Dice and—in the 3D evaluation—gamma results and comparison figures. DICOM converters copy geometry from a reference RTDOSE and replace the dose grid/scaling. The reference RTDOSE must belong to the same patient/study/frame of reference.

Each `predict_mimicking_parameters.py` builds a ten-channel volume:

```text
[normalized CT, 8 ROI masks, normalized predicted dose]
```

It resizes this volume to the requested `(D,H,W)` and applies a dual-head 3D ResNet. The CSV columns are:

| Column | Meaning |
|---|---|
| `TOW` | recommended Target-versus-OAR weight ratio |
| `VDP` | recommended voxel dose priority |

The implementation applies `abs()` and rounding (`TOW`: 2 decimals, `VDP`: 1 decimal). It does not clamp values to a RayStation-supported range. Review every prediction and validate the mapping against the RayStation version and institutional mimicking protocol.

## 4. RayStation

Left (`LT`) and right (`RT`) workflows are separated. For each side, scripts are numbered in execution order:

1. `01_*_2D_angle.py`, `02_*_25D_angle.py`, or `03_*_3D_angle.py`: normalize ROI metadata, create a five-fraction VMAT plan, place isocenter and create two arcs using the model-specific angle table.
2. `04_*_COPY_*_angle.py`: export the mimicked beam-set dose and copy the plan for optimization.
3. `05_*_OPT_REV.py`: load clinical goals, create/adjust objectives, optimize iteratively, scale prescription and export results.

The optimization script uses an institutional clinical-goal template (`b_dm_g`), machine (`ELT33V`), imaging system (`Canon_EXLB_2022`), ROI names and 26 Gy/5-fraction assumptions. These are configuration requirements, not universal defaults. Test on a non-clinical database first.

