import os
import glob
import re
import time
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
import pydicom
from pydicom.uid import generate_uid, RTDoseStorage
from pydicom.dataset import FileMetaDataset

# =========================
# 기본 설정
# =========================
working_dir = "../"  # 예측 NPY가 들어있는 상위 경로
dicom_input_root = "./WBI_LT"  # 원본 DICOM(root)
dicom_output_path = "DICOM_OUTPUT"

prescription_dose = 26.00
target_value = prescription_dose * 1.1  # 네트워크 예측 → 실제 Gy 로 스케일링할 때 사용
noise_threshold_ratio = 0.025  # max dose 의 2.5% 미만은 0으로 컷

# =========================
# 유틸 함수들
# =========================

def extract_patient_id_from_string(name_str: str) -> str:
    """
    '098WL', '098', '098 AAA' 등의 문자열에서 앞쪽 숫자만 추출.
    못 찾으면 ValueError.
    """
    s = str(name_str)
    m = re.search(r"(\d+)", s)
    if not m:
        raise ValueError(f"환자번호를 찾을 수 없습니다: {s}")
    return m.group(1)


def extract_patient_id_from_dicom(ds: pydicom.Dataset) -> str:
    """
    DICOM Dataset에서 PatientName 우선, 안되면 PatientID에서 숫자를 추출.
    """
    for tag in ["PatientName", "PatientID"]:
        if hasattr(ds, tag):
            try:
                return extract_patient_id_from_string(getattr(ds, tag))
            except ValueError:
                continue
    raise ValueError("DICOM에서 환자번호를 찾지 못했습니다.")


def extract_patient_id_from_npy_filename(filename: str) -> str:
    """
    NPY 파일명에서 환자 번호를 추출.
    기존 코드의 npy_file.split('_')[4] 로직을 유지하되, 예외 시 fallback.
    """
    parts = filename.split('_')
    if len(parts) > 4:
        m = re.match(r"(\d+)", parts[4])
        if m:
            return m.group(1)

    # fallback: 파일명 전체에서 첫 숫자 시퀀스를 찾기
    return extract_patient_id_from_string(filename)


def extract_origin_from_npy_filename(filename: str):
    """
    파일명 안의 'origin' 토큰 뒤에 오는 3개의 값을 (x, y, z)로 사용.
    예: ..._origin_-255.51171875_-468.51171875_-460.0.npy_pred_dose.npy
    """
    parts = filename.split('_')
    if "origin" not in parts:
        raise ValueError(f"'origin' 토큰을 찾을 수 없습니다: {filename}")

    idx = parts.index("origin")
    try:
        x_str = parts[idx + 1]
        y_str = parts[idx + 2]
        z_str = parts[idx + 3]  # 보통 '-460.0.npy' 같은 형태
    except IndexError:
        raise ValueError(f"'origin' 뒤의 좌표 3개를 읽을 수 없습니다: {filename}")

    # 숫자만 추출해서 float으로 변환
    def _to_float(s: str) -> float:
        m = re.search(r"[-+]?\d*\.?\d+([eE][-+]?\d+)?", s)
        if not m:
            raise ValueError(f"좌표에서 숫자를 추출할 수 없습니다: {s}")
        return float(m.group(0))

    x = _to_float(x_str)
    y = _to_float(y_str)
    z = _to_float(z_str)

    return x, y, z


def build_patient_dicom_map(dicom_root: str):
    """
    ./WBI_LT/*/*_CT_*, *_RTst_*, *_RTDOSE_* 폴더를 모두 검색해서
    patient_id → {"CT": ..., "RTST": ..., "RTDOSE": ...} 형태의 딕셔너리 생성.
    """
    patient_dicoms = defaultdict(dict)

    ct_folders = sorted(glob.glob(os.path.join(dicom_root, "*/*_CT_*")))
    rtst_folders = sorted(glob.glob(os.path.join(dicom_root, "*/*_RTst_*")))
    rtdose_folders = sorted(glob.glob(os.path.join(dicom_root, "*/*_RTDOSE_*")))

    print(f"[INFO] CT  폴더 수 : {len(ct_folders)}")
    print(f"[INFO] RTst 폴더 수 : {len(rtst_folders)}")
    print(f"[INFO] RTDOSE 폴더 수 : {len(rtdose_folders)}")

    def _scan_folder_list(folder_list, modality_label):
        for folder in folder_list:
            dcm_files = sorted(os.listdir(folder))
            if not dcm_files:
                continue
            first_dcm = os.path.join(folder, dcm_files[0])
            ds = pydicom.dcmread(first_dcm, stop_before_pixels=True, force=True)
            pid = extract_patient_id_from_dicom(ds)
            patient_dicoms[pid][modality_label] = folder

    _scan_folder_list(ct_folders, "CT")
    _scan_folder_list(rtst_folders, "RTST")
    _scan_folder_list(rtdose_folders, "RTDOSE")

    print(f"[INFO] DICOM이 매핑된 환자 수: {len(patient_dicoms)}명")
    return patient_dicoms


# def build_prediction_index(working_dir: str):
#     """
#     ../*/*prediction*/ 폴더들 안의 *.npy 파일을 모두 스캔해서
#     model_type → patient_id → [npy_paths...] 인덱스를 구축.
#     """
#     prediction_folder_list = sorted(glob.glob(os.path.join(working_dir, "*/*prediction*/")))
#     print(f"[INFO] prediction 폴더 수: {len(prediction_folder_list)}")

#     index = defaultdict(lambda: defaultdict(list))

#     for pred_dir in prediction_folder_list:
#         # 상위 상위 폴더명을 model_type으로 사용 (기존 코드와 동일한 정의)
#         model_type = os.path.basename(os.path.dirname(os.path.dirname(pred_dir)))
#         npy_files = sorted(glob.glob(os.path.join(pred_dir, "*.npy")))
#         print(f"  - Model: {model_type}, NPY 파일 수: {len(npy_files)} (폴더: {pred_dir})")
#         for npy_path in npy_files:
#             fname = os.path.basename(npy_path)
#             patient_id = extract_patient_id_from_npy_filename(fname)
#             index[model_type][patient_id].append(npy_path)

#     model_types = sorted(index.keys())
#     print(f"[INFO] 발견된 model_type: {model_types}")
#     return model_types, index

def build_prediction_index(working_dir: str):
    """
    기존: working_dir/*/*prediction*/ 만 찾았기 때문에
    3D 폴더(3.unet_3D_...)는 스캔되지 않았음.

    수정: 
    - 2D/2.5D는 기존처럼 prediction 폴더를 사용
    - 3D는 현재 폴더의 '3.unet_3D_REV_256_MSE_fetching_processing_agument_apply'
      자체를 prediction 폴더로 취급
    """

    index = defaultdict(lambda: defaultdict(list))

    # 1) 기존 2D/2.5D prediction 폴더
    prediction_folder_list = sorted(
        glob.glob(os.path.join(working_dir, "*/*prediction*/"))
    )

    # 2) 3D 폴더(현재 디렉토리)에 직접 있는 경우 추가
    three_d_dir = "3.unet_3D_REV_256_MSE_fetching_processing_agument_apply"
    if os.path.isdir(three_d_dir):
        prediction_folder_list.append(three_d_dir)

    print(f"[INFO] prediction 폴더 수: {len(prediction_folder_list)}")

    for pred_dir in prediction_folder_list:
        pred_dir = os.path.normpath(pred_dir)

        # 3D 폴더인지 판단: 이름이 3.unet_3D... 로 시작하고, 바로 그 폴더에 npy가 있음
        if os.path.basename(pred_dir).startswith("3.unet_3D"):
            model_type = os.path.basename(pred_dir)  # 폴더명 그대로 사용
            npy_files = sorted(glob.glob(os.path.join(pred_dir, "*.npy")))
        else:
            # 기존 2D/2.5D: 상위 폴더명이 모델명
            model_type = os.path.basename(os.path.dirname(pred_dir))
            npy_files = sorted(glob.glob(os.path.join(pred_dir, "*.npy")))

        print(f"  - Model: {model_type}, NPY 파일 수: {len(npy_files)} (폴더: {pred_dir})")

        for npy_path in npy_files:
            fname = os.path.basename(npy_path)
            patient_id = extract_patient_id_from_npy_filename(fname)
            index[model_type][patient_id].append(npy_path)

    model_types = sorted(index.keys())
    print(f"[INFO] 발견된 model_type: {model_types}")
    return model_types, index



def copy_original_dicoms_for_patient(model_type: str,
                                     patient_id: str,
                                     patient_dicoms: dict,
                                     dicom_output_root: str):
    """
    한 model_type, 한 patient에 대해 원본 CT/RTST/RTDOSE 폴더를
    DICOM_OUTPUT/model_type/patient_id 아래로 복사.
    이미 있으면 건너뜀.
    """
    dest_root = os.path.join(dicom_output_root, model_type, patient_id)
    os.makedirs(dest_root, exist_ok=True)

    dicom_info = patient_dicoms.get(patient_id, {})
    for modality, src_folder in dicom_info.items():
        folder_name = os.path.basename(src_folder)
        dest_folder = os.path.join(dest_root, folder_name)
        if os.path.exists(dest_folder) and os.listdir(dest_folder):
            # 이미 복사된 것으로 간주
            continue
        print(f"    [COPY] {modality}: {src_folder} -> {dest_folder}")
        shutil.copytree(src_folder, dest_folder, dirs_exist_ok=True)


def load_ct_reference_dicom(ct_folder: str) -> pydicom.Dataset:
    """
    CT 폴더에서 InstanceNumber가 가장 작은 슬라이스를 찾아
    reference CT DICOM으로 사용.
    """
    dcm_files = sorted(os.listdir(ct_folder))
    if not dcm_files:
        raise RuntimeError(f"CT 폴더가 비어 있습니다: {ct_folder}")

    best_ds = None
    best_instance = None
    for fname in dcm_files:
        path = os.path.join(ct_folder, fname)
        ds = pydicom.dcmread(path, force=True)
        inst = getattr(ds, "InstanceNumber", None)
        if inst is None:
            continue
        if best_instance is None or inst < best_instance:
            best_instance = inst
            best_ds = ds

    if best_ds is None:
        # InstanceNumber가 없으면 그냥 첫 파일 사용
        best_ds = pydicom.dcmread(os.path.join(ct_folder, dcm_files[0]), force=True)
    return best_ds


def create_rtdose_dataset(ct_ds: pydicom.Dataset,
                          dose_array_gy: np.ndarray,
                          origin_xyz,
                          prescription_dose: float):
    """
    CT DICOM과 실제 Gy 스케일의 dose_array (shape: [Z, H, W])를 받아
    RTDOSE DICOM Dataset 생성.
    origin_xyz: (x, y, z) in mm (ImagePositionPatient)
    """
    z_dim, h_dim, w_dim = dose_array_gy.shape

    # ── 기본 DICOM 헤더 설정 ─────────────────
    ds = pydicom.Dataset()
    ds.PatientName = getattr(ct_ds, "PatientName", "")
    ds.PatientID = getattr(ct_ds, "PatientID", "")
    ds.StudyInstanceUID = getattr(ct_ds, "StudyInstanceUID", generate_uid())
    ds.FrameOfReferenceUID = getattr(ct_ds, "FrameOfReferenceUID", generate_uid())
    ds.Modality = "RTDOSE"
    ds.StudyDate = getattr(ct_ds, "StudyDate", "")
    ds.StudyTime = getattr(ct_ds, "StudyTime", "")
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = generate_uid()
    ds.SOPClassUID = RTDoseStorage

    # Dose 관련 태그
    ds.DoseUnits = "GY"
    ds.DoseType = "PHYSICAL"
    ds.DoseSummationType = "PLAN"

    # ── Geometry ─────────────────────────────
    ds.Rows = h_dim
    ds.Columns = w_dim
    ds.NumberOfFrames = z_dim

    # 여기서는 2.0mm spacing 가정 (기존 코드 유지)
    grid_spacing = 2.0
    ds.PixelSpacing = [str(grid_spacing), str(grid_spacing)]
    ds.SliceThickness = str(grid_spacing)
    ds.GridFrameOffsetVector = [i * grid_spacing for i in range(z_dim)]

    ds.ImageOrientationPatient = getattr(ct_ds, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])
    x, y, z = origin_xyz
    ds.ImagePositionPatient = [float(x), float(y), float(z)]

    # ── DoseGridScaling & PixelData ──────────
    max_physical_dose = prescription_dose * 1.1
    max_pixel_value = np.iinfo(np.uint16).max
    dose_grid_scaling = max_physical_dose / max_pixel_value
    ds.DoseGridScaling = dose_grid_scaling

    # 범위를 넘어가면 saturate
    dose_clipped = np.clip(dose_array_gy, 0, max_physical_dose)
    scaled_dose = np.round(dose_clipped / dose_grid_scaling).astype(np.uint16)
    ds.PixelData = scaled_dose.tobytes()

    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    # FrameIncrementPointer
    ds.FrameIncrementPointer = pydicom.tag.Tag(0x3004, 0x000C)

    # CT StudyInstanceUID와 동기화
    ds[0x0020000D] = ct_ds[0x0020000D]

    # 파일 메타 정보
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    return ds


def convert_single_prediction(model_type: str,
                              patient_id: str,
                              npy_path: str,
                              ct_ds: pydicom.Dataset,
                              dicom_output_root: str,
                              prescription_dose: float):
    """
    하나의 (model_type, patient_id, npy_path)에 대해
    NPY → RTDOSE DICOM 변환을 수행하고, 변환 시간(sec)과 프레임 수를 반환.
    """
    fname = os.path.basename(npy_path)
    print(f"    [CONVERT] Model={model_type}, Patient={patient_id}, NPY={fname}")

    # 시간 측정 시작
    t0 = time.perf_counter()

    # 1. NPY 로드 및 스케일링
    arr = np.load(npy_path)  # 예: (H, W, Z)
    if arr.ndim != 3:
        raise ValueError(f"예상치 못한 dose array shape (3D가 아님): {arr.shape} @ {npy_path}")

    # (H, W, Z) → (Z, H, W)
    dose_array = np.transpose(arr, (2, 0, 1))
    dose_array_gy = dose_array * target_value

    # 2. 낮은 선량 노이즈 제거
    thr = noise_threshold_ratio * np.max(dose_array_gy)
    dose_array_gy[dose_array_gy < thr] = 0.0

    # 3. origin 좌표 추출
    origin_xyz = extract_origin_from_npy_filename(fname)

    # 4. RTDOSE Dataset 생성
    rtdose = create_rtdose_dataset(ct_ds, dose_array_gy, origin_xyz, prescription_dose)

    # 5. 출력 경로 구성 & 저장
    patient_root = os.path.join(dicom_output_root, model_type, patient_id)
    save_dir = os.path.join(patient_root, f"{patient_id}_RTDOSE_PREDICTION")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{patient_id}_RTDOSE.dcm")

    pydicom.dcmwrite(out_path, rtdose, write_like_original=False)

    elapsed = time.perf_counter() - t0
    num_frames = dose_array_gy.shape[0]
    print(f"      → 저장 완료: {out_path} (frames={num_frames}, time={elapsed:.3f}s)")
    return elapsed, num_frames, out_path


# =========================
# 메인 루틴
# =========================

def main():
    # 1) 원본 DICOM 매핑 (환자번호 → CT/RTst/RTDOSE 폴더)
    patient_dicoms = build_patient_dicom_map(dicom_input_root)

    # 2) prediction 폴더 인덱싱 (model_type → patient_id → [npy_paths...])
    model_types, pred_index = build_prediction_index(working_dir)

    os.makedirs(dicom_output_path, exist_ok=True)

    # 변환 로그 (시간 분석용)
    conversion_records = []

    # 3) 모델 타입 / 환자별로 변환 진행
    for model_type in model_types:
        patients = pred_index[model_type]
        model_root = os.path.join(dicom_output_path, model_type)
        os.makedirs(model_root, exist_ok=True)

        print(f"\n[MODEL] {model_type} (환자 수: {len(patients)})")
        for patient_id, npy_list in patients.items():
            if patient_id not in patient_dicoms or "CT" not in patient_dicoms[patient_id]:
                print(f"  [WARN] CT DICOM을 찾을 수 없어 스킵: Patient {patient_id}")
                continue

            # 3-1) 원본 DICOM 복사 (CT/RTST/RTDOSE)
            copy_original_dicoms_for_patient(model_type, patient_id, patient_dicoms, dicom_output_path)

            # 3-2) CT reference DICOM 로드
            ct_folder = patient_dicoms[patient_id]["CT"]
            ct_ds = load_ct_reference_dicom(ct_folder)

            # 3-3) 이 환자에 대해 첫 번째 NPY만 사용 (여러 개면 경고 출력)
            npy_list_sorted = sorted(npy_list)
            if len(npy_list_sorted) > 1:
                print(f"  [INFO] Patient {patient_id}에 대해 NPY가 {len(npy_list_sorted)}개 발견, 첫 번째만 사용합니다.")

            npy_path = npy_list_sorted[0]

            # 3-4) NPY → RTDOSE
            elapsed, num_frames, out_path = convert_single_prediction(
                model_type=model_type,
                patient_id=patient_id,
                npy_path=npy_path,
                ct_ds=ct_ds,
                dicom_output_root=dicom_output_path,
                prescription_dose=prescription_dose
            )

            conversion_records.append(
                {
                    "model_type": model_type,
                    "patient_id": patient_id,
                    "npy_file": os.path.basename(npy_path),
                    "num_frames": num_frames,
                    "elapsed_sec": elapsed,
                    "output_dicom": out_path,
                }
            )

    # 4) 시간 분석 결과 요약
    if conversion_records:
        df = pd.DataFrame(conversion_records)
        print("\n================ NPY → DICOM 변환 시간 요약 ================")
        print(df.head())

        # ---- (1) 전체 raw 기록 저장 ----
        time_csv = "time_summary.csv"
        df.to_csv(time_csv, index=False)
        print(f"\n[INFO] 변환 로그 CSV 저장 완료: {time_csv}")

        # ---- (2) 모델별 통계 저장 ----
        stats_by_model = df.groupby("model_type")["elapsed_sec"].describe()
        print("\n[모델별 변환 시간 통계 (초)]")
        print(stats_by_model)

        stats_by_model.to_csv("time_stats_by_model.csv")
        print("[INFO] 모델별 통계 CSV 저장 완료: time_stats_by_model.csv")

        # ---- (3) 환자별 평균 시간 저장 ----
        stats_by_patient = (
            df.groupby("patient_id")["elapsed_sec"]
              .mean()
              .reset_index()
              .rename(columns={"elapsed_sec": "mean_elapsed_sec"})
        )
        print("\n[환자별 평균 변환 시간 (초)]")
        print(stats_by_patient.sort_values("mean_elapsed_sec"))

        stats_by_patient.to_csv("time_stats_by_patient.csv", index=False)
        print("[INFO] 환자별 통계 CSV 저장 완료: time_stats_by_patient.csv")

    else:
        print("\n[INFO] 변환된 RTDOSE가 없습니다. prediction 폴더 및 NPY 경로를 확인하세요.")



if __name__ == "__main__":
    main()
