# Configuration and privacy

The published defaults use repository-relative paths and de-identified output. Copy `config.example.env` values into the process environment used for training or RayStation. The project deliberately does not load a committed `.env` file because it may contain local paths or sensitive values.

## Path variables

| Variable | Default | Purpose |
|---|---|---|
| `WBI_DATA_ROOT` | `./data` | Train/Validation/Test array root |
| `WBI_OUTPUT_ROOT` | `./outputs` | Predictions, evaluations and figures |
| `WBI_PREDICTION_ROOT` | model-specific folder under `WBI_OUTPUT_ROOT` | Predicted NPY input for RTDOSE conversion |
| `WBI_DICOM_INPUT_ROOT` | `./data/dicom` | Source DICOM root |
| `WBI_DICOM_OUTPUT_ROOT` | `./outputs/dicom` | Generated RTDOSE root |
| `WBI_EXPORT_ROOT` | `C:\WBI_exports` | RayStation DICOM export root |

## DICOM identity variables

`WBI_ANONYMIZE_DICOM=true` makes the Python RTDOSE converters replace Patient Name and Patient ID with `WBI_ANONYMIZED_NAME` and `WBI_ANONYMIZED_ID`. If `WBI_ANONYMIZED_ID` is empty, a stable hash-derived case ID is generated. Source CT/RTSTRUCT/RTDOSE copying is disabled by default with `WBI_COPY_SOURCE_DICOM=false`; enabling it may copy identifiable DICOM unchanged. `WBI_RAYSTATION_ANONYMIZE=true` controls RayStation scripted export. These are safe defaults but are not a complete DICOM de-identification guarantee: private tags, dates, descriptions, burned-in pixels and UID policy still require an institution-approved de-identification tool and review.

Do not set either variable to `false` outside an approved clinical system. Do not put real identifiers in this repository, configuration examples, filenames, screenshots, logs or Git history.

## RayStation site variables

Set `WBI_RAYSTATION_MACHINE`, `WBI_RAYSTATION_IMAGING_SYSTEM` and `WBI_CLINICAL_GOAL_TEMPLATE` to commissioned local values. Copy `04_raystation/angles.example.json` outside the repository, populate locally approved angle mappings and set `WBI_ANGLE_CONFIG` to that file. The real angle table is intentionally not versioned because it may encode cohort/case identifiers.

RayStation export folders use a SHA-256-derived case label by default rather than Patient Name. Patient identifiers are omitted from optimization logs unless `WBI_LOG_PATIENT_IDENTIFIERS=true` is explicitly set.

## Shell example

```bash
export WBI_DATA_ROOT=/approved/deidentified/wbi/data
export WBI_OUTPUT_ROOT=/approved/deidentified/wbi/outputs
export WBI_DICOM_INPUT_ROOT=/approved/deidentified/wbi/dicom
export WBI_DICOM_OUTPUT_ROOT=/approved/deidentified/wbi/rtdose
export WBI_ANONYMIZE_DICOM=true
export WBI_COPY_SOURCE_DICOM=false
```
