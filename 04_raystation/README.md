# RayStation integration

These scripts must be run from RayStation's scripting environment; a normal CPython installation does not provide `connect` or RayStation objects.

Before use, configure the export root, commissioned machine and imaging-system names, clinical-goal template, ROI names, prescription/fractions, dose algorithm, arc angles and plan-selection assumptions. Paths and site identifiers are read from the environment variables documented in [`../docs/CONFIGURATION.md`](../docs/CONFIGURATION.md). Copy `angles.example.json` outside the repository, populate an approved local table and point `WBI_ANGLE_CONFIG` to it. No cohort-specific angle table is included in the scripts.

DICOM export defaults to anonymization, export folders use a hash-derived case label, and patient names/IDs are omitted from optimization logs. Disable those safeguards only under an approved clinical data-handling procedure.

Import the predicted RTDOSE only after confirming Patient ID, Study/Series/Frame of Reference UIDs, voxel geometry, orientation and dose units. In dose mimicking, enter the case's predicted `TOW` and `VDP` from `final_prediction.csv`; the scripts do not automatically ingest that CSV. Perform independent dose calculation, clinical-goal review and qualified-person approval before any clinical use.

The scripts are research automation and are not a substitute for RayStation commissioning, local QA, or clinical judgment.
