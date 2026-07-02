# Model weights

Dose-prediction checkpoints smaller than GitHub's per-file limit are stored beside each model in `02_training/*/weights/`.

The four parameter-regression checkpoints are about 255 MB each and are intentionally excluded from Git history. Download the matching checkpoint from the GitHub Release named `model-weights` and save it as:

```text
weights/parameter_models/
├── 2d_best_model.pth
├── 25d_best_model.pth
└── 3d_best_model.pth
```

The two 2.5D experiment folders contained byte-identical parameter checkpoints, so one shared asset is published. Verify the SHA-256 checksums published with the Release before inference. Checkpoints are research artifacts, not independently validated medical devices.
