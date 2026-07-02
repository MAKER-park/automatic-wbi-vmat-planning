import os
import numpy as np
import pydicom
import importlib.util
import sys
from tqdm import tqdm

# Import helper functions from reference code
def import_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

ref_script_path = "referece_code/inference_and_evaluation/2.npy_DICOM_convert.py"
if not os.path.exists(ref_script_path):
    # Fallback if the path is relative to root
    ref_script_path = "../referece_code/inference_and_evaluation/2.npy_DICOM_convert.py"

try:
    ref_module = import_from_file("dicom_convert", ref_script_path)
    build_patient_dicom_map = ref_module.build_patient_dicom_map
    extract_origin_from_npy_filename = ref_module.extract_origin_from_npy_filename
    load_ct_reference_dicom = ref_module.load_ct_reference_dicom
    create_rtdose_dataset = ref_module.create_rtdose_dataset
    extract_patient_key_from_npy_filename = ref_module.extract_patient_key_from_npy_filename
    copy_original_dicoms_for_patient = ref_module.copy_original_dicoms_for_patient
except Exception as e:
    print(f"Error importing reference module: {e}")
    sys.exit(1)

# --- Configuration ---
PRED_NPY_DIR = os.path.join(os.environ.get("WBI_OUTPUT_ROOT", "./outputs"), "prediction_npy_3d_hd_unet_stage1")
DICOM_INPUT_ROOT = os.environ.get("WBI_DICOM_INPUT_ROOT", "./data/dicom")
DICOM_OUTPUT_ROOT = os.path.join(os.environ.get("WBI_DICOM_OUTPUT_ROOT", "./outputs/dicom"), "DICOM_OUTPUT_3D_stage1")
PRESCRIPTION_DOSE = 26.0
TARGET_VALUE = PRESCRIPTION_DOSE * 1.1

def main():
    os.makedirs(DICOM_OUTPUT_ROOT, exist_ok=True)
    
    # 1. 원본 DICOM 매핑 (환자번호 → CT/RTst/RTDOSE 폴더)
    print("Building patient DICOM map...")
    patient_dicoms = build_patient_dicom_map(DICOM_INPUT_ROOT)
    
    # 2. 예측 파일 순회
    if not os.path.exists(PRED_NPY_DIR):
        print(f"Error: {PRED_NPY_DIR} not found.")
        return
        
    npy_files = sorted([f for f in os.listdir(PRED_NPY_DIR) if f.endswith('.npy')])
    print(f"Found {len(npy_files)} prediction files.")

    for fname in tqdm(npy_files, desc="Converting NPY to DICOM"):
        try:
            patient_id = extract_patient_key_from_npy_filename(fname)
            if patient_id not in patient_dicoms or "CT" not in patient_dicoms[patient_id]:
                print(f"  [WARN] CT DICOM not found for {patient_id}. Skipping.")
                continue
            
            # 3. 원본 DICOM 복사 (필요시)
            copy_original_dicoms_for_patient("HD-3D-Unet", patient_id, patient_dicoms, DICOM_OUTPUT_ROOT)
            
            # 4. CT reference DICOM 로드
            ct_folder = patient_dicoms[patient_id]["CT"]
            ct_ds = load_ct_reference_dicom(ct_folder)
            
            # 5. NPY 로드 (H, W, Z) -> (Z, H, W)
            dose_arr = np.load(os.path.join(PRED_NPY_DIR, fname))
            dose_arr_gy = np.transpose(dose_arr, (2, 0, 1)) * TARGET_VALUE
            
            # 6. origin 좌표 추출
            origin_xyz = extract_origin_from_npy_filename(fname)
            
            # 7. RTDOSE Dataset 생성
            rtdose = create_rtdose_dataset(
                ct_ds=ct_ds,
                dose_array_gy=dose_arr_gy,
                origin_xyz=origin_xyz,
                prescription_dose=PRESCRIPTION_DOSE,
                model_type="HD-3D-Unet",
                prediction_label="3d_pred"
            )
            
            # 8. 저장
            patient_root = os.path.join(DICOM_OUTPUT_ROOT, "HD-3D-Unet", patient_id)
            save_dir = os.path.join(patient_root, f"{patient_id}_RTDOSE_3D_PRED")
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"{patient_id}_RTDOSE_3D.dcm")
            
            pydicom.dcmwrite(out_path, rtdose, write_like_original=False)
            
        except Exception as e:
            print(f"  [ERROR] Failed to process {fname}: {e}")

    print(f"NPY to DICOM conversion complete. Results in {DICOM_OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
