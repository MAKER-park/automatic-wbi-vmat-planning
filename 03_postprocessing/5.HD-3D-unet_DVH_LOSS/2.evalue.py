import os
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wilcoxon, shapiro, ttest_rel
from scipy.ndimage import zoom
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# 경고 메시지 무시
warnings.filterwarnings("ignore", category=UserWarning, module='scipy')
warnings.filterwarnings("ignore", category=FutureWarning, module='seaborn')


# --- Configuration ---
# 경로 설정
PRED_DIR = "prediction_npy_3d_hd_unet"
GT_BASE_DIR = "../final_dataset/Test"
GT_DOSE_DIR = os.path.join(GT_BASE_DIR, "Dose")
CT_CONTOUR_DIR = os.path.join(GT_BASE_DIR, "CT_and_Contour")
VIS_OUTPUT_DIR = "evaluation_visuals_3D" # 시각화 결과 저장 폴더

# train_agu.py 전역 변수 참조
RX_DOSE = 26.0
TARGET_DOSE = RX_DOSE * 1.10 # 110% 스케일링 -> 28.6 Gy -> 105% -> 27.3 Gy 
SUPERSAMPLE_FACTOR = 1 # DVH 계산을 위한 업샘플링 계수 (train_agu.py와 일치)

# ROI 이름과 CT/Contour 파일 내 채널 인덱스 매핑
ROI_MAP = {
    "CTV": 2,
    "Heart": 5,
    "Ipsi_Lung": 6,
    "Contra_Lung": 7,
    "Contra_Breast": 3,
    "External": 4, # External(Body) 마스크 추가
}
# DVH 그래프 색상 설정
ROI_COLORS = {
    "CTV": "red",
    "Heart": "magenta",
    "Ipsi_Lung": "blue",
    "Contra_Lung": "cyan",
    "Contra_Breast": "green",
    "External": "gray",
}


# --- Metric Calculation Functions ---

def get_statistical_p_value(gt_series, pred_series):
    """정규성 검정 후 적절한 통계 검정 수행"""
    if len(gt_series) < 3 or len(pred_series) < 3:
        return np.nan, "N/A"
    
    diff = gt_series - pred_series
    
    # 데이터가 모두 동일한 경우 처리
    if np.all(diff == 0):
        return 1.0, "Identical"

    # Shapiro-Wilk 정규성 검정 (p > 0.05 이면 정규분포 가정)
    try:
        _, p_norm = shapiro(diff)
    except:
        p_norm = 0.0 # 에러 발생 시 비모수 검정으로 유도

    if p_norm > 0.05:
        # 정규 분포를 따를 때: 대응표본 t-검정
        p_val = ttest_rel(gt_series, pred_series).pvalue
        method = "T-test"
    else:
        # 정규 분포를 따르지 않을 때: 윌콕슨 부호 순위 검정
        p_val = wilcoxon(gt_series, pred_series).pvalue
        method = "Wilcoxon"
        
    return p_val, method

def calculate_dice_3d(true_vol, pred_vol, threshold):
    """특정 임계값 이상을 1로 간주하여 3D DICE 점수 계산"""
    true_mask = true_vol >= threshold
    pred_mask = pred_vol >= threshold
    
    intersection = np.sum(true_mask * pred_mask)
    total = np.sum(true_mask) + np.sum(pred_mask)
    
    if total == 0:
        return 1.0
        
    return (2. * intersection) / total

def calculate_ssim_3d(true_vol, pred_vol, data_range, mask=None):
    """3D SSIM 점수 계산 (마스크가 제공되면 해당 영역 내에서만 계산)"""
    if mask is not None and np.sum(mask) > 0:
        # full=True를 사용하여 SSIM 맵을 가져옴
        _, ssim_map = ssim(true_vol, pred_vol, data_range=data_range, full=True)
        return np.mean(ssim_map[mask > 0])
    return ssim(true_vol, pred_vol, data_range=data_range)

def calculate_mae_3d(true_vol, pred_vol, mask=None):
    """3D MAE (Mean Absolute Error) 계산"""
    if mask is not None and np.sum(mask) > 0:
        return np.mean(np.abs(true_vol[mask > 0] - pred_vol[mask > 0]))
    return np.mean(np.abs(true_vol - pred_vol))

def get_dvh_metrics(dose_vol_gy, roi_masks):
    """주어진 3D 선량 볼륨과 ROI 마스크에 대해 DVH 관련 지표 계산"""
    metrics = {}
    for name, mask in roi_masks.items():
        if np.sum(mask) == 0:
            continue
            
        dose_in_roi = dose_vol_gy[mask]
        
        metrics[f"{name}_Dmax"] = np.max(dose_in_roi)
        metrics[f"{name}_Dmean"] = np.mean(dose_in_roi)
        
        if name == 'CTV':
            v95_threshold = RX_DOSE * 0.95
            metrics['CTV_V95%'] = (np.sum(dose_in_roi >= v95_threshold) / len(dose_in_roi)) * 100.0
        
        if name == 'Heart':
            metrics['Heart_V7Gy'] = (np.sum(dose_in_roi >= 7.0) / len(dose_in_roi)) * 100.0
            metrics['Heart_V1.5Gy'] = (np.sum(dose_in_roi >= 1.5) / len(dose_in_roi)) * 100.0
        if name == 'Ipsi_Lung':
            metrics['Ipsi_Lung_V8Gy'] = (np.sum(dose_in_roi >= 8.0) / len(dose_in_roi)) * 100.0
        
    return metrics

# --- Visualization Functions ---

def dvh_curve_cumulative(dose_gy_3d, mask_3d, bins_gy):
    """train_agu.py에서 가져온 DVH 계산 함수"""
    vals = dose_gy_3d[mask_3d > 0]
    if vals.size == 0:
        return None
    v = np.array([(vals >= d).mean() * 100.0 for d in bins_gy], dtype=np.float32)
    return v

def save_dvh_figure(true_gy_3d, pred_gy_3d, roi_vols_3d, out_path):
    """DVH 그래프를 생성하고 저장"""
    bins_gy = np.linspace(0, TARGET_DOSE * 1.1, 200)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    for roi_name, mask3d in roi_vols_3d.items():
        color = ROI_COLORS.get(roi_name, "black")
        v_true = dvh_curve_cumulative(true_gy_3d, mask3d, bins_gy)
        v_pred = dvh_curve_cumulative(pred_gy_3d, mask3d, bins_gy)
        
        if v_true is not None:
            ax.plot(bins_gy, v_true, label=f"{roi_name} (True)", linestyle='-', color=color)
        if v_pred is not None:
            ax.plot(bins_gy, v_pred, label=f"{roi_name} (Pred)", linestyle='--', color=color)

    ax.set_xlabel("Dose (Gy)")
    ax.set_ylabel("Volume (%)")
    ax.set_title("DVH Comparison (True: solid, Pred: dashed)")
    ax.set_xlim(0, bins_gy[-1])
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

def save_comparison_figure(ct_slice, roi_dict_2d, true_dose_slice, pred_dose_slice, out_path):
    """4분할 비교 이미지를 생성하고 저장"""
    diff_slice = true_dose_slice - pred_dose_slice
    vmax = TARGET_DOSE
    diff_vmax = np.max(np.abs(diff_slice))

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 1. CT + ROI Contours
    axes[0, 0].imshow(ct_slice, cmap="gray", vmin=-200, vmax=200)
    for name, mask in roi_dict_2d.items():
        if np.sum(mask) > 0:
            axes[0, 0].contour(mask, colors=[ROI_COLORS.get(name, 'white')], levels=[0.5], linewidths=1)
    axes[0, 0].set_title("CT with ROI Contours")
    axes[0, 0].axis("off")

    # 2. True Dose
    im1 = axes[0, 1].imshow(true_dose_slice, cmap='jet', vmin=0, vmax=vmax)
    axes[0, 1].set_title("Ground Truth Dose")
    axes[0, 1].axis("off")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # 3. Predicted Dose
    im2 = axes[1, 0].imshow(pred_dose_slice, cmap='jet', vmin=0, vmax=vmax)
    axes[1, 0].set_title("Predicted Dose")
    axes[1, 0].axis("off")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 4. Difference
    im3 = axes[1, 1].imshow(diff_slice, cmap='bwr', vmin=-diff_vmax, vmax=diff_vmax)
    axes[1, 1].set_title("Difference (True - Pred)")
    axes[1, 1].axis("off")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

def save_box_plots(df, group_name, out_dir):
    """그룹별 모든 지표에 대한 box plot 생성 및 저장"""
    gt_cols = sorted([col for col in df.columns if '_gt' in col])
    pred_cols = sorted([col for col in df.columns if '_pred' in col])

    for gt_col, pred_col in zip(gt_cols, pred_cols):
        metric_name = gt_col.replace('_gt', '')
        
        # p-value 계산
        gt_series = df[gt_col].dropna()
        pred_series = df[pred_col].dropna()
        p_value = wilcoxon(gt_series, pred_series).pvalue if len(gt_series) > 1 and len(pred_series) > 1 else 1.0

        # 데이터프레임 재구성
        plot_df = pd.DataFrame({
            'value': pd.concat([gt_series, pred_series]),
            'type': ['Ground Truth'] * len(gt_series) + ['Prediction'] * len(pred_series)
        })

        plt.figure(figsize=(8, 6))
        sns.boxplot(x='type', y='value', data=plot_df)
        plt.title(f'{group_name}: {metric_name}\n(p-value: {p_value:.4f})')
        plt.xlabel('')
        plt.ylabel(metric_name)
        plt.tight_layout()
        
        # 파일 이름으로 부적합한 문자 제거
        safe_metric_name = metric_name.replace('%', 'pct').replace('/', '')
        plt.savefig(os.path.join(out_dir, f"boxplot_{safe_metric_name}.png"))
        plt.close()

# --- Main Analysis Function ---

def analyze_and_print_results(results_list, group_name, md_file_handle, vis_dir):
    """결과 분석, 출력, Readme.md 작성 및 Box plot 저장"""
    if not results_list:
        print(f"\n--- No results found for group: {group_name} ---")
        md_file_handle.write(f"\n## Evaluation Results for {group_name} (n=0)\n")
        md_file_handle.write("No results found.\n")
        return

    df = pd.DataFrame(results_list).set_index('patient_id')
    
    gt_cols = sorted([col for col in df.columns if '_gt' in col])
    pred_cols = sorted([col for col in df.columns if '_pred' in col])

    analysis_summary = []
    for gt_col, pred_col in zip(gt_cols, pred_cols):
        metric_name = gt_col.replace('_gt', '')
        gt_series, pred_series = df[gt_col].dropna(), df[pred_col].dropna()
        
        p_value, method = get_statistical_p_value(gt_series, pred_series)

        analysis_summary.append({
            "Metric": metric_name,
            "GT (mean ± std)": f"{gt_series.mean():.2f} ± {gt_series.std():.2f}",
            "Pred (mean ± std)": f"{pred_series.mean():.2f} ± {pred_series.std():.2f}",
            "p-value": f"{p_value:.4f}" if not np.isnan(p_value) else "N/A",
            "Method": method
        })

    summary_df = pd.DataFrame(analysis_summary)
    
    # 콘솔 출력
    print(f"\n--- Evaluation Results for {group_name} (n={len(df)}) ---")
    print(summary_df.to_string(index=False))

    # Readme.md 파일에 쓰기
    md_file_handle.write(f"\n## Evaluation Results for {group_name} (n={len(df)})\n")
    md_file_handle.write(summary_df.to_markdown(index=False))
    md_file_handle.write("\n")

    # Box plot 저장
    group_vis_dir = os.path.join(vis_dir, group_name)
    os.makedirs(group_vis_dir, exist_ok=True)
    save_box_plots(df, group_name, group_vis_dir)


def main():
    """메인 실행 함수"""
    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)
    
    pred_files = [f for f in os.listdir(PRED_DIR) if f.endswith('_pred_dose.npy')]
    if not pred_files:
        print(f"Error: No prediction files found in '{PRED_DIR}'")
        return

    results_wl, results_wr = [], []

    for pred_file in tqdm(pred_files, desc="Evaluating predictions"):
        original_ct_contour_file = pred_file.replace('_pred_dose.npy', '.npy')
        ct_contour_path = os.path.join(CT_CONTOUR_DIR, original_ct_contour_file)
        gt_dose_file = original_ct_contour_file.replace('_CT_and_Contour_', '_Dose_')
        gt_dose_path = os.path.join(GT_DOSE_DIR, gt_dose_file)

        if not (os.path.exists(gt_dose_path) and os.path.exists(ct_contour_path)):
            continue

        # 원본 해상도 데이터 로드
        pred_dose_norm = np.load(os.path.join(PRED_DIR, pred_file))
        gt_dose_raw = np.load(gt_dose_path)
        ct_contour_4d = np.load(ct_contour_path)

        pred_dose_gy_orig = np.transpose(pred_dose_norm, (2, 0, 1)) * TARGET_DOSE
        gt_dose_gy_orig = np.transpose(gt_dose_raw, (2, 0, 1)) * TARGET_DOSE
        roi_masks_orig = {name: np.transpose(ct_contour_4d[:, :, :, idx] > 0.5, (2, 0, 1)) for name, idx in ROI_MAP.items()}

        patient_id = original_ct_contour_file.replace('.npy', '')
        patient_results = {'patient_id': patient_id}
        
        # --- 1. 공간적 일치도 (DICE) ---
        patient_results['DICE_95%_gt'] = 1.0
        patient_results['DICE_95%_pred'] = calculate_dice_3d(gt_dose_gy_orig, pred_dose_gy_orig, RX_DOSE * 0.95)
        
        patient_results['DICE_50%_gt'] = 1.0
        patient_results['DICE_50%_pred'] = calculate_dice_3d(gt_dose_gy_orig, pred_dose_gy_orig, RX_DOSE * 0.50)
        
        # --- 2. 선량 차이 (MAE, Gy) ---
        patient_results['MAE_Total_gt'] = 0.0
        patient_results['MAE_Total_pred'] = calculate_mae_3d(gt_dose_gy_orig, pred_dose_gy_orig)
        
        if 'External' in roi_masks_orig:
            patient_results['MAE_Exter_gt'] = 0.0
            patient_results['MAE_Exter_pred'] = calculate_mae_3d(gt_dose_gy_orig, pred_dose_gy_orig, mask=roi_masks_orig['External'])
            
        if 'CTV' in roi_masks_orig:
            patient_results['MAE_CTV_gt'] = 0.0
            patient_results['MAE_CTV_pred'] = calculate_mae_3d(gt_dose_gy_orig, pred_dose_gy_orig, mask=roi_masks_orig['CTV'])

        # --- 3. 구조적 유사도 (SSIM) ---
        patient_results['SSIM_Total_gt'] = 1.0
        patient_results['SSIM_Total_pred'] = calculate_ssim_3d(gt_dose_gy_orig, pred_dose_gy_orig, data_range=TARGET_DOSE)
        
        if 'External' in roi_masks_orig:
            patient_results['SSIM_Exter_gt'] = 1.0
            patient_results['SSIM_Exter_pred'] = calculate_ssim_3d(gt_dose_gy_orig, pred_dose_gy_orig, data_range=TARGET_DOSE, mask=roi_masks_orig['External'])

        if 'CTV' in roi_masks_orig:
            patient_results['SSIM_CTV_gt'] = 1.0
            patient_results['SSIM_CTV_pred'] = calculate_ssim_3d(gt_dose_gy_orig, pred_dose_gy_orig, data_range=TARGET_DOSE, mask=roi_masks_orig['CTV'])

        # --- DVH 계산을 위한 업샘플링 ---
        if SUPERSAMPLE_FACTOR > 1:
            pred_dose_gy_dvh = zoom(pred_dose_gy_orig, SUPERSAMPLE_FACTOR, order=0) # order=1 (선형 보간) -> order=0 (최근접 이웃 보간)으로 변경
            gt_dose_gy_dvh = zoom(gt_dose_gy_orig, SUPERSAMPLE_FACTOR, order=0)   # order=1 (선형 보간) -> order=0 (최근접 이웃 보간)으로 변경
            roi_masks_dvh = {
                name: zoom(mask.astype(float), SUPERSAMPLE_FACTOR, order=1) > 0.5 # order 0: 최근접 이웃 보간 order 1: 선형 보간
                for name, mask in roi_masks_orig.items()
            }
        else:
            pred_dose_gy_dvh = pred_dose_gy_orig
            gt_dose_gy_dvh = gt_dose_gy_orig
            roi_masks_dvh = roi_masks_orig

        # DVH 관련 지표는 업샘플링된 데이터로 계산
        dvh_gt = get_dvh_metrics(gt_dose_gy_dvh, roi_masks_dvh)
        dvh_pred = get_dvh_metrics(pred_dose_gy_dvh, roi_masks_dvh)
        for key, val in dvh_gt.items(): patient_results[f"{key}_gt"] = val
        for key, val in dvh_pred.items(): patient_results[f"{key}_pred"] = val
            
        if "WL" in patient_id:
            results_wl.append(patient_results)
        elif "WR" in patient_id:
            results_wr.append(patient_results)
        else:
            results_wr.append(patient_results)

        # --- 환자별 시각화 ---
        patient_vis_dir = os.path.join(VIS_OUTPUT_DIR, patient_id)
        os.makedirs(patient_vis_dir, exist_ok=True)

        # 1. DVH Figure (업샘플링된 데이터 사용)
        save_dvh_figure(gt_dose_gy_dvh, pred_dose_gy_dvh, roi_masks_dvh, os.path.join(patient_vis_dir, "dvh_comparison.png"))

        # 2. 4-panel Figure (원본 해상도 데이터 사용)
        z_mid = gt_dose_gy_orig.shape[0] // 2
        ct_slice = np.transpose(ct_contour_4d, (2, 0, 1, 3))[z_mid, :, :, 0]
        roi_dict_2d = {name: mask[z_mid] for name, mask in roi_masks_orig.items()}
        save_comparison_figure(ct_slice, roi_dict_2d, gt_dose_gy_orig[z_mid], pred_dose_gy_orig[z_mid], os.path.join(patient_vis_dir, "dose_comparison.png"))

    # --- 상세 결과 저장 (CSV) ---
    all_results = results_wl + results_wr
    if all_results:
        df_all = pd.DataFrame(all_results)
        # 컬럼 순서 정렬 (patient_id를 맨 앞으로)
        cols = ['patient_id'] + sorted([c for c in df_all.columns if c != 'patient_id'])
        df_all = df_all[cols]
        df_all.to_csv("case_dvh_evaluation_result.csv", index=False)
        print(f"Detailed case results saved to 'case_dvh_evaluation_result.csv'.")

    # --- 그룹 분석 및 Readme.md 생성 ---
    with open("inference_result_3d.md", "w") as md_file:
        md_file.write("# Dose Prediction Evaluation Report\n")
        analyze_and_print_results(results_wl, "WL (Whole Left)", md_file, VIS_OUTPUT_DIR)
        analyze_and_print_results(results_wr, "WR (Whole Right)", md_file, VIS_OUTPUT_DIR)
    
    print(f"\nEvaluation complete. All visualizations saved in '{VIS_OUTPUT_DIR}' directory.")
    print(f"Summary report saved to 'inference_result_3d.md'.")


if __name__ == '__main__':
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', 5)
    pd.set_option('display.width', 120)
    main()
