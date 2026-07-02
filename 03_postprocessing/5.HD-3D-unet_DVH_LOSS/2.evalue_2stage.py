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
from numba import njit, prange
from joblib import Parallel, delayed
import multiprocessing

# 경고 메시지 무시
warnings.filterwarnings("ignore", category=UserWarning, module='scipy')
warnings.filterwarnings("ignore", category=FutureWarning, module='seaborn')


# --- Configuration ---
# 경로 설정
PRED_DIR = "prediction_npy_3d_hd_unet_stage2"
GT_BASE_DIR = "../final_dataset/Test"
GT_DOSE_DIR = os.path.join(GT_BASE_DIR, "Dose")
CT_CONTOUR_DIR = os.path.join(GT_BASE_DIR, "CT_and_Contour")
VIS_OUTPUT_DIR = "evaluation_visuals_3D_2stage" # 시각화 결과 저장 폴더

# train_agu.py 전역 변수 참조
RX_DOSE = 26.0
TARGET_DOSE = RX_DOSE * 1.10 # 110% 스케일링 -> 28.6 Gy
SUPERSAMPLE_FACTOR = 1 # DVH 계산을 위한 업샘플링 계수 (train_agu.py와 일치)

# Gamma Index 설정
VOXEL_SIZE = (3.0, 1.0, 1.0) # (dz, dy, dx)
GAMMA_DOSE_THRESH_PCT = 10.0 # 10% threshold

# ROI 이름과 CT/Contour 파일 내 채널 인덱스 매핑
ROI_MAP = {
    "CTV": 2,
    "Heart": 5,
    "Ipsi_Lung": 6,
    "Contra_Lung": 7,
    "Contra_Breast": 3,
    "External": 4, 
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


# --- Numba Optimized Gamma Functions ---

@njit(parallel=True)
def _calculate_gamma_3d_numba(ref, evaluation, mask, delta_dose, delta_dist, voxel_size, search_radius_vox):
    dz, dy, dx = ref.shape
    gamma_map = np.ones((dz, dy, dx), dtype=np.float32) * 5.0
    
    delta_dose_sq = delta_dose**2
    delta_dist_sq = delta_dist**2
    vs_z, vs_y, vs_x = voxel_size
    sr_z, sr_y, sr_x = search_radius_vox

    for z in prange(dz):
        for y in range(dy):
            for x in range(dx):
                if not mask[z, y, x]:
                    continue
                
                min_gamma_sq = 1e10
                
                z_min = max(0, z - sr_z)
                z_max = min(dz, z + sr_z + 1)
                y_min = max(0, y - sr_y)
                y_max = min(dy, y + sr_y + 1)
                x_min = max(0, x - sr_x)
                x_max = min(dx, x + sr_x + 1)
                
                ref_val = ref[z, y, x]
                
                for sz in range(z_min, z_max):
                    for sy in range(y_min, y_max):
                        for sx in range(x_min, x_max):
                            dist_sq = ((z - sz) * vs_z)**2 + \
                                      ((y - sy) * vs_y)**2 + \
                                      ((x - sx) * vs_x)**2
                            
                            dose_diff_sq = (ref_val - evaluation[sz, sy, sx])**2
                            gamma_sq = (dist_sq / delta_dist_sq) + (dose_diff_sq / delta_dose_sq)
                            
                            if gamma_sq < min_gamma_sq:
                                min_gamma_sq = gamma_sq
                
                gamma_map[z, y, x] = np.sqrt(min_gamma_sq)
    
    return gamma_map

@njit(parallel=True)
def _calculate_gamma_2d_numba(ref, evaluation, mask, delta_dose, delta_dist, pixel_spacing, search_radius_vox):
    dy, dx = ref.shape
    gamma_map = np.ones((dy, dx), dtype=np.float32) * 5.0
    
    delta_dose_sq = delta_dose**2
    delta_dist_sq = delta_dist**2
    vs_y, vs_x = pixel_spacing
    sr_y, sr_x = search_radius_vox

    for y in prange(dy):
        for x in range(dx):
            if not mask[y, x]:
                continue
            
            min_gamma_sq = 1e10
            
            y_min = max(0, y - sr_y)
            y_max = min(dy, y + sr_y + 1)
            x_min = max(0, x - sr_x)
            x_max = min(dx, x + sr_x + 1)
            
            ref_val = ref[y, x]
            
            for sy in range(y_min, y_max):
                for sx in range(x_min, x_max):
                    dist_sq = ((y - sy) * vs_y)**2 + \
                              ((x - sx) * vs_x)**2
                    
                    dose_diff_sq = (ref_val - evaluation[sy, sx])**2
                    gamma_sq = (dist_sq / delta_dist_sq) + (dose_diff_sq / delta_dose_sq)
                    
                    if gamma_sq < min_gamma_sq:
                        min_gamma_sq = gamma_sq
            
            gamma_map[y, x] = np.sqrt(min_gamma_sq)
    
    return gamma_map

def calculate_gamma_index(ref, evaluation, dose_crit_pct, dist_crit_mm, voxel_size, 
                          dose_threshold_pct=10.0, is_relative=True):
    ref_max = np.max(ref)
    if ref_max == 0: return np.zeros_like(ref), 0.0
    mask = ref >= (ref_max * dose_threshold_pct / 100.0)
    if np.sum(mask) == 0: return np.zeros_like(ref), 0.0

    delta_dose = (dose_crit_pct / 100.0) * (ref_max if is_relative else RX_DOSE)
    delta_dist = dist_crit_mm
    search_radius_vox = np.array([int(np.ceil(delta_dist / vs)) for vs in voxel_size])
    
    gamma_map = _calculate_gamma_3d_numba(ref.astype(np.float32), evaluation.astype(np.float32), 
                                         mask, delta_dose, delta_dist, np.array(voxel_size), search_radius_vox)
    
    pass_rate = (np.sum(gamma_map[mask] <= 1.0) / np.sum(mask)) * 100.0
    return gamma_map, pass_rate

def calculate_gamma_index_2d(ref_slice, eval_slice, dose_crit_pct, dist_crit_mm, pixel_spacing, 
                             dose_threshold_pct=10.0, is_relative=True):
    ref_max = np.max(ref_slice)
    if ref_max == 0: return np.zeros_like(ref_slice), 0.0
    mask = ref_slice >= (ref_max * dose_threshold_pct / 100.0)
    if np.sum(mask) == 0: return np.zeros_like(ref_slice), 0.0

    delta_dose = (dose_crit_pct / 100.0) * (ref_max if is_relative else RX_DOSE)
    delta_dist = dist_crit_mm
    search_radius_vox = np.array([int(np.ceil(delta_dist / ps)) for ps in pixel_spacing])
    
    gamma_map = _calculate_gamma_2d_numba(ref_slice.astype(np.float32), eval_slice.astype(np.float32), 
                                         mask, delta_dose, delta_dist, np.array(pixel_spacing), search_radius_vox)
    
    pass_rate = (np.sum(gamma_map[mask] <= 1.0) / np.sum(mask)) * 100.0
    return gamma_map, pass_rate

# --- Metric Calculation Functions ---

def get_statistical_p_value(gt_series, pred_series):
    if len(gt_series) < 3 or len(pred_series) < 3: return np.nan, "N/A"
    diff = gt_series - pred_series
    if np.all(diff == 0): return 1.0, "Identical"
    try:
        _, p_norm = shapiro(diff)
    except:
        p_norm = 0.0
    if p_norm > 0.05:
        p_val = ttest_rel(gt_series, pred_series).pvalue
        method = "T-test"
    else:
        p_val = wilcoxon(gt_series, pred_series).pvalue
        method = "Wilcoxon"
    return p_val, method

def calculate_dice_3d(true_vol, pred_vol, threshold):
    true_mask, pred_mask = true_vol >= threshold, pred_vol >= threshold
    intersection = np.sum(true_mask * pred_mask)
    total = np.sum(true_mask) + np.sum(pred_mask)
    return (2. * intersection) / total if total > 0 else 1.0

def calculate_ssim_3d(true_vol, pred_vol, data_range, mask=None):
    from skimage.metrics import structural_similarity as ssim_func
    if mask is not None and np.sum(mask) > 0:
        _, ssim_map = ssim_func(true_vol, pred_vol, data_range=data_range, full=True)
        return np.mean(ssim_map[mask > 0])
    return ssim_func(true_vol, pred_vol, data_range=data_range)

def calculate_mae_3d(true_vol, pred_vol, mask=None):
    if mask is not None and np.sum(mask) > 0:
        return np.mean(np.abs(true_vol[mask > 0] - pred_vol[mask > 0]))
    return np.mean(np.abs(true_vol - pred_vol))

def get_dvh_metrics(dose_vol_gy, roi_masks):
    metrics = {}
    for name, mask in roi_masks.items():
        if np.sum(mask) == 0: continue
        dose_in_roi = dose_vol_gy[mask]
        metrics[f"{name}_Dmax"] = np.max(dose_in_roi)
        metrics[f"{name}_Dmean"] = np.mean(dose_in_roi)
        if name == 'CTV':
            metrics['CTV_V95%'] = (np.sum(dose_in_roi >= RX_DOSE * 0.95) / len(dose_in_roi)) * 100.0
        if name == 'Heart':
            metrics['Heart_V7Gy'] = (np.sum(dose_in_roi >= 7.0) / len(dose_in_roi)) * 100.0
            metrics['Heart_V1.5Gy'] = (np.sum(dose_in_roi >= 1.5) / len(dose_in_roi)) * 100.0
        if name == 'Ipsi_Lung':
            metrics['Ipsi_Lung_V8Gy'] = (np.sum(dose_in_roi >= 8.0) / len(dose_in_roi)) * 100.0
    return metrics

# --- Visualization Functions ---

def dvh_curve_cumulative(dose_gy_3d, mask_3d, bins_gy):
    vals = dose_gy_3d[mask_3d > 0]
    if vals.size == 0: return None
    return np.array([(vals >= d).mean() * 100.0 for d in bins_gy], dtype=np.float32)

def save_dvh_figure(true_gy_3d, pred_gy_3d, roi_vols_3d, out_path):
    bins_gy = np.linspace(0, TARGET_DOSE * 1.1, 200)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for roi_name, mask3d in roi_vols_3d.items():
        color = ROI_COLORS.get(roi_name, "black")
        v_true = dvh_curve_cumulative(true_gy_3d, mask3d, bins_gy)
        v_pred = dvh_curve_cumulative(pred_gy_3d, mask3d, bins_gy)
        if v_true is not None: ax.plot(bins_gy, v_true, label=f"{roi_name} (True)", linestyle='-', color=color)
        if v_pred is not None: ax.plot(bins_gy, v_pred, label=f"{roi_name} (Pred)", linestyle='--', color=color)
    ax.set_xlabel("Dose (Gy)"); ax.set_ylabel("Volume (%)"); ax.set_title("DVH Comparison")
    ax.set_xlim(0, bins_gy[-1]); ax.set_ylim(0, 100); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); plt.savefig(out_path); plt.close(fig)

def save_comparison_figure(ct_slice, roi_dict_2d, true_dose_slice, pred_dose_slice, gamma_map_2d, out_path):
    diff_slice = true_dose_slice - pred_dose_slice
    diff_vmax = np.max(np.abs(diff_slice))
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes[0, 0].imshow(ct_slice, cmap="gray", vmin=-200, vmax=200)
    for name, mask in roi_dict_2d.items():
        if np.sum(mask) > 0: axes[0, 0].contour(mask, colors=[ROI_COLORS.get(name, 'white')], levels=[0.5], linewidths=1)
    axes[0, 0].set_title("CT with ROI Contours"); axes[0, 0].axis("off")
    im1 = axes[0, 1].imshow(true_dose_slice, cmap='jet', vmin=0, vmax=TARGET_DOSE); axes[0, 1].set_title("GT Dose"); axes[0, 1].axis("off")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    im2 = axes[0, 2].imshow(pred_dose_slice, cmap='jet', vmin=0, vmax=TARGET_DOSE); axes[0, 2].set_title("Pred Dose"); axes[0, 2].axis("off")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
    im3 = axes[1, 0].imshow(diff_slice, cmap='bwr', vmin=-diff_vmax, vmax=diff_vmax); axes[1, 0].set_title("Diff (True-Pred)"); axes[1, 0].axis("off")
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    im4 = axes[1, 1].imshow(gamma_map_2d, cmap='RdYlGn_r', vmin=0, vmax=2); axes[1, 1].set_title("Gamma (3%/3mm, Rel)"); axes[1, 1].axis("off")
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    axes[1, 2].axis("off"); plt.tight_layout(); plt.savefig(out_path); plt.close(fig)

# --- Processing Functions ---

def process_single_patient(pred_file):
    original_ct_contour_file = pred_file.replace('_pred_dose.npy', '.npy')
    ct_contour_path = os.path.join(CT_CONTOUR_DIR, original_ct_contour_file)
    gt_dose_file = original_ct_contour_file.replace('_CT_and_Contour_', '_Dose_')
    gt_dose_path = os.path.join(GT_DOSE_DIR, gt_dose_file)

    if not (os.path.exists(gt_dose_path) and os.path.exists(ct_contour_path)):
        return None

    # Load data
    pred_dose_norm = np.load(os.path.join(PRED_DIR, pred_file))
    gt_dose_raw = np.load(gt_dose_path)
    ct_contour_4d = np.load(ct_contour_path)

    pred_dose_gy_orig = np.transpose(pred_dose_norm, (2, 0, 1)) * TARGET_DOSE
    gt_dose_gy_orig = np.transpose(gt_dose_raw, (2, 0, 1)) * TARGET_DOSE
    roi_masks_orig = {name: np.transpose(ct_contour_4d[:, :, :, idx] > 0.5, (2, 0, 1)) for name, idx in ROI_MAP.items()}

    patient_id = original_ct_contour_file.replace('.npy', '')
    res = {'patient_id': patient_id}
    
    # Standard metrics
    res['DICE_95%_gt'], res['DICE_95%_pred'] = 1.0, calculate_dice_3d(gt_dose_gy_orig, pred_dose_gy_orig, RX_DOSE * 0.95)
    res['MAE_Total_gt'], res['MAE_Total_pred'] = 0.0, calculate_mae_3d(gt_dose_gy_orig, pred_dose_gy_orig)
    res['SSIM_Total_gt'], res['SSIM_Total_pred'] = 1.0, calculate_ssim_3d(gt_dose_gy_orig, pred_dose_gy_orig, TARGET_DOSE)

    # Gamma Index (3D)
    for rel_mode, suffix in [(True, "Rel"), (False, "Abs")]:
        for c in [1, 2, 3]:
            _, pass_rate = calculate_gamma_index(gt_dose_gy_orig, pred_dose_gy_orig, c, c, VOXEL_SIZE, GAMMA_DOSE_THRESH_PCT, rel_mode)
            res[f'Gamma_3D_{suffix}_{c}%{c}mm'] = pass_rate

    # Gamma Index (2D central slice)
    z_mid = gt_dose_gy_orig.shape[0] // 2
    gt_slice, pred_slice = gt_dose_gy_orig[z_mid], pred_dose_gy_orig[z_mid]
    
    g_map_2d_vis = None
    for rel_mode, suffix in [(True, "Rel"), (False, "Abs")]:
        for c in [1, 2, 3]:
            g_map, pass_rate = calculate_gamma_index_2d(gt_slice, pred_slice, c, c, VOXEL_SIZE[1:], GAMMA_DOSE_THRESH_PCT, rel_mode)
            res[f'Gamma_2D_{suffix}_{c}%{c}mm'] = pass_rate
            if suffix == "Rel" and c == 3: g_map_2d_vis = g_map

    # DVH
    dvh_gt = get_dvh_metrics(gt_dose_gy_orig, roi_masks_orig)
    dvh_pred = get_dvh_metrics(pred_dose_gy_orig, roi_masks_orig)
    for k, v in dvh_gt.items(): res[f"{k}_gt"] = v
    for k, v in dvh_pred.items(): res[f"{k}_pred"] = v
            
    # Visuals
    p_vis_dir = os.path.join(VIS_OUTPUT_DIR, patient_id); os.makedirs(p_vis_dir, exist_ok=True)
    save_dvh_figure(gt_dose_gy_orig, pred_dose_gy_orig, roi_masks_orig, os.path.join(p_vis_dir, "dvh_comparison.png"))
    ct_slice = np.transpose(ct_contour_4d, (2, 0, 1, 3))[z_mid, :, :, 0]
    save_comparison_figure(ct_slice, {n: m[z_mid] for n, m in roi_masks_orig.items()}, gt_slice, pred_slice, g_map_2d_vis, os.path.join(p_vis_dir, "dose_comparison_with_gamma.png"))

    return res

def main():
    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)
    pred_files = [f for f in os.listdir(PRED_DIR) if f.endswith('_pred_dose.npy')]
    if not pred_files: return

    # Numba JIT 컴파일 강제 수행 (첫 환자 처리 전 워밍업)
    print("Compiling optimized functions (Numba JIT)...")
    dummy = np.zeros((4, 4, 4), dtype=np.float32)
    _calculate_gamma_3d_numba(dummy, dummy, dummy>0, 1.0, 1.0, np.array([1.0, 1.0, 1.0]), np.array([1, 1, 1]))

    # 병렬 처리 실행
    num_cores = multiprocessing.cpu_count()
    print(f"Starting parallel evaluation on {num_cores} cores...")
    results = Parallel(n_jobs=num_cores)(delayed(process_single_patient)(f) for f in tqdm(pred_files))
    results = [r for r in results if r is not None]

    # 결과 분류 및 분석
    results_wl = [r for r in results if "WL" in r['patient_id']]
    results_wr = [r for r in results if "WR" in r['patient_id'] or ("WL" not in r['patient_id'] and "WR" not in r['patient_id'])]

    df_all = pd.DataFrame(results)
    cols = ['patient_id'] + sorted([c for c in df_all.columns if c != 'patient_id'])
    df_all[cols].to_csv("case_dvh_evaluation_result_2stage.csv", index=False)

    with open("inference_result_3d_2stage.md", "w") as md_file:
        md_file.write("# Dose Prediction Evaluation Report (Optimized)\n")
        analyze_and_print_results(results_wl, "WL (Whole Left)", md_file, VIS_OUTPUT_DIR)
        analyze_and_print_results(results_wr, "WR (Whole Right)", md_file, VIS_OUTPUT_DIR)
    
    print(f"\nEvaluation complete. Report saved to 'inference_result_3d_2stage.md'.")

def analyze_and_print_results(results_list, group_name, md_file_handle, vis_dir):
    if not results_list: return
    df = pd.DataFrame(results_list).set_index('patient_id')
    paired = sorted(list(set([c.replace('_gt', '').replace('_pred', '') for c in df.columns if '_gt' in c or '_pred' in c])))
    single = sorted([c for c in df.columns if '_gt' not in c and '_pred' not in c])
    summary = []
    for m in paired:
        g, p = f"{m}_gt", f"{m}_pred"
        if g in df.columns and p in df.columns:
            gv, pv = df[g].dropna(), df[p].dropna()
            pval, meth = get_statistical_p_value(gv, pv)
            summary.append({"Metric": m, "GT (mean±std)": f"{gv.mean():.2f}±{gv.std():.2f}", "Pred (mean±std)": f"{pv.mean():.2f}±{pv.std():.2f}", "p-value": f"{pval:.4f}" if not np.isnan(pval) else "N/A"})
    for m in single:
        v = df[m].dropna()
        summary.append({"Metric": m, "GT (mean±std)": "N/A", "Pred (mean±std)": f"{v.mean():.2f}±{v.std():.2f}", "p-value": "N/A"})
    summary_df = pd.DataFrame(summary)
    print(f"\n--- {group_name} ---"); print(summary_df.to_string(index=False))
    md_file_handle.write(f"\n## {group_name}\n" + summary_df.to_markdown(index=False) + "\n")
    g_dir = os.path.join(vis_dir, group_name); os.makedirs(g_dir, exist_ok=True)
    # Box plots (Skipped for brevity in summary, but would be here)

if __name__ == '__main__':
    main()
