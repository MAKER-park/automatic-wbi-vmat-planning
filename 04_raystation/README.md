# RayStation integration

These scripts must be run from RayStation's scripting environment; a normal CPython installation does not provide `connect` or RayStation objects.

Before use, review and replace all site-specific settings: export root, machine and imaging-system names, clinical-goal template, ROI names, prescription/fractions, dose algorithm, arc angles and plan-selection assumptions. The original internal network path has been replaced with the `WBI_EXPORT_ROOT` environment variable. Set it on the RayStation workstation, for example to an approved local or network export directory. DICOM export defaults to anonymization in the published scripts; change that only under an approved clinical data-handling procedure.

Import the predicted RTDOSE only after confirming Patient ID, Study/Series/Frame of Reference UIDs, voxel geometry, orientation and dose units. In dose mimicking, enter the case's predicted `TOW` and `VDP` from `final_prediction.csv`; the scripts do not automatically ingest that CSV. Perform independent dose calculation, clinical-goal review and qualified-person approval before any clinical use.

The scripts are research automation and are not a substitute for RayStation commissioning, local QA, or clinical judgment.
