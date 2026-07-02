# inference_matched_no_csv.py
# ------------------------------------------------------------
# CSV 없이 실행:
# - pred_dir 안의 prediction npy들을 스캔
# - 파일명 매칭 로직(치환 -> pid 후보+유사도)으로 CT_and_Contour(+옵션: GT Dose) 찾기
# - 10채널 입력을 런타임에 조립: [CT(1) + ROI masks(8) + Dose(1)]
#   * dose_source="pred": 10번째 채널에 pred_dose 사용 (보통 실사용 추론에 권장)
#   * dose_source="gt"  : 10번째 채널에 GT Dose 사용 (학습 때 GT dose를 넣었으면 이걸로)
# - 결과를 CSV로 저장: inference_details.csv, final_prediction.csv, missing_matches.csv
# ------------------------------------------------------------

import os
import re
import difflib
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --------- Settings (training과 동일하게 유지) ----------
clip_min = -300
clip_max = 800
model_type = "3D_"


# ============================================================
# 1) Matching utilities (평가 코드와 동일한 로직)
# ============================================================
_PAT_PID = re.compile(r"_(\d{3}(?:WL|WR))_")

def extract_patient_id(fname: str):
    m = _PAT_PID.search(fname)
    return m.group(1) if m else None

def safe_replace_pred_to_gt(pred_fname: str, gt_kind: str):
    """
    Replace '_pred_dose_' (case-insensitive) with '_CT_and_Contour_' or '_Dose_'.
    """
    if gt_kind not in ("CT_and_Contour", "Dose"):
        raise ValueError("gt_kind must be 'CT_and_Contour' or 'Dose'")
    return re.sub(r"(?i)_pred_dose_", f"_{gt_kind}_", pred_fname)

def build_index(folder: str):
    """
    Build {pid: [filenames...]} for all .npy files in a folder.
    pid is extracted by regex _###WL_ or _###WR_.
    """
    idx = {}
    if not os.path.exists(folder):
        return idx
    for fn in os.listdir(folder):
        if not fn.endswith(".npy"):
            continue
        pid = extract_patient_id(fn)
        if pid:
            idx.setdefault(pid, []).append(fn)
    return idx

def best_match_by_similarity(target_name: str, candidates: list[str]):
    """
    Choose the candidate filename with max difflib SequenceMatcher ratio.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    best = None
    best_score = -1.0
    for c in candidates:
        score = difflib.SequenceMatcher(a=target_name, b=c).ratio()
        if score > best_score:
            best_score = score
            best = c
    return best

def resolve_paths_from_pred_name(
    pred_name: str,
    pred_dir: str,
    ct_dir: str,
    dose_dir: str,
    ct_index: dict,
    dose_index: dict,
    need_gt_dose: bool,
):
    """
    pred_name: pred_dir 내 파일명 (basename)
    Returns (pred_path, ct_path, gt_dose_path_or_None, ct_pick_name, dose_pick_name)
    Matching order:
      1) filename replace: _Pred_Dose_ -> _CT_and_Contour_ / _Dose_
      2) fallback: pid -> candidates -> best string similarity
    """
    pred_path = os.path.join(pred_dir, pred_name)
    if not os.path.exists(pred_path):
        return None, None, None, None, None

    # 1) rule-based replacement
    ct_name_try = safe_replace_pred_to_gt(pred_name, "CT_and_Contour")
    ct_path_try = os.path.join(ct_dir, ct_name_try)

    dose_name_try = None
    dose_path_try = None
    if need_gt_dose:
        dose_name_try = safe_replace_pred_to_gt(pred_name, "Dose")
        dose_path_try = os.path.join(dose_dir, dose_name_try)

    if os.path.exists(ct_path_try) and ((not need_gt_dose) or os.path.exists(dose_path_try)):
        return pred_path, ct_path_try, dose_path_try, os.path.basename(ct_path_try), (os.path.basename(dose_path_try) if dose_path_try else None)

    # 2) fallback by pid and similarity
    pid = extract_patient_id(pred_name)
    if pid is None:
        return None, None, None, None, None

    ct_candidates = ct_index.get(pid, [])
    ct_pick = best_match_by_similarity(pred_name, ct_candidates)
    if ct_pick is None:
        return None, None, None, None, None
    ct_path = os.path.join(ct_dir, ct_pick)

    gt_dose_path = None
    dose_pick = None
    if need_gt_dose:
        dose_candidates = dose_index.get(pid, [])
        dose_pick = best_match_by_similarity(pred_name, dose_candidates)
        if dose_pick is None:
            return None, None, None, None, None
        gt_dose_path = os.path.join(dose_dir, dose_pick)

    if os.path.exists(ct_path) and ((not need_gt_dose) or os.path.exists(gt_dose_path)):
        return pred_path, ct_path, gt_dose_path, os.path.basename(ct_path), (os.path.basename(gt_dose_path) if gt_dose_path else None)

    return None, None, None, None, None


# ============================================================
# 2) I/O + preprocessing (10채널 조립)
# ============================================================
def _ctcontour_to_cdhw(ct4d: np.ndarray) -> np.ndarray:
    """
    Expect (H,W,D,C). Return (C,D,H,W).
    """
    ct4d = np.asarray(ct4d)
    if ct4d.ndim != 4:
        raise ValueError(f"CT_and_Contour must be 4D (H,W,D,C); got shape={ct4d.shape}")
    return np.transpose(ct4d, (3, 2, 0, 1))

def _dose_to_dhw(arr: np.ndarray) -> np.ndarray:
    """
    Accept (H,W,D) or (H,W,D,1) etc -> squeeze -> (H,W,D) then -> (D,H,W)
    """
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Dose must become 3D after squeeze; got shape={arr.shape}")
    return np.transpose(arr, (2, 0, 1))

def build_10ch_tensor(
    pred_path: str,
    ct_path: str,
    gt_dose_path: str | None,
    img_size: tuple[int, int, int],
    dose_source: str,
) -> torch.Tensor:
    """
    Returns torch.FloatTensor with shape (10, D', H', W') after resize.
    dose_source:
      - "pred": uses pred_dose file for 10th channel
      - "gt"  : uses gt_dose file for 10th channel (gt_dose_path required)
    """
    pred_dose = np.load(pred_path)         # expected (H,W,D) (or squeezeable)
    ct_contour_4d = np.load(ct_path)       # expected (H,W,D,C=9)

    ct_cdhw = _ctcontour_to_cdhw(ct_contour_4d)  # (C,D,H,W)
    if ct_cdhw.shape[0] < 9:
        raise ValueError(f"CT_and_Contour channels < 9. Got {ct_cdhw.shape[0]} channels from: {ct_path}")

    ct_channel = ct_cdhw[0]    # (D,H,W)
    masks = ct_cdhw[1:9]       # (8,D,H,W)

    if dose_source == "gt":
        if gt_dose_path is None:
            raise ValueError("dose_source='gt' requires gt_dose_path")
        gt_dose = np.load(gt_dose_path)
        dose_dhw = _dose_to_dhw(gt_dose)
    else:
        dose_dhw = _dose_to_dhw(pred_dose)

    # --- Normalization (기존 inference 코드 유지) ---
    ct_channel = np.clip(ct_channel, clip_min, clip_max)
    ct_channel = (ct_channel - clip_min) / (clip_max - clip_min)

    d_min, d_max = float(dose_dhw.min()), float(dose_dhw.max())
    if d_max > d_min:
        dose_dhw = (dose_dhw - d_min) / (d_max - d_min)
    else:
        dose_dhw = np.zeros_like(dose_dhw)

    processed = np.concatenate(
        [ct_channel[np.newaxis, ...], masks, dose_dhw[np.newaxis, ...]],
        axis=0
    ).astype(np.float32)  # (10,D,H,W)

    x = torch.from_numpy(processed)  # (10,D,H,W)
    x = F.interpolate(
        x.unsqueeze(0),
        size=img_size,            # (D,H,W)
        mode="trilinear",
        align_corners=False
    ).squeeze(0)
    return x


# ============================================================
# 3) Model (사용자 코드와 동일)
# ============================================================
class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out

class DualHeadResNet3D(nn.Module):
    def __init__(self, block=BasicBlock3D, num_blocks=[3, 4, 6, 3], in_channels=10):
        super(DualHeadResNet3D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc_intensity = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        self.fc_smoothness = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        features = torch.flatten(out, 1)
        return self.fc_intensity(features).squeeze(), self.fc_smoothness(features).squeeze()


def _load_state_dict_safely(model, model_path, device):
    ckpt = torch.load(model_path, map_location=device)

    # case1: {"state_dict": ...}
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
    # case2: pure state_dict
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise RuntimeError("Unsupported checkpoint format (expected dict).")

    # strip "module." if trained with DDP
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    model.load_state_dict(sd, strict=True)


# ============================================================
# 4) Main inference (CSV 없이 pred_dir 스캔)
# ============================================================
def list_pred_files(pred_dir: str, mode: str):
    """
    mode:
      - "eval_like": *_dose_aligned.npy and contains 'pred_dose' (case-insensitive)
      - "all_npy": all .npy in pred_dir
    """
    files = []
    for f in os.listdir(pred_dir):
        if not f.endswith(".npy"):
            continue
        fl = f.lower()
        if mode == "eval_like":
            if fl.endswith("_dose_aligned.npy") and ("pred_dose" in fl):
                files.append(f)
        else:
            files.append(f)
    files.sort()
    return files

def run_inference(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Inference] Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.pred_dir):
        print(f"[Error] pred_dir not found: {args.pred_dir}")
        return
    if not os.path.exists(args.ct_dir):
        print(f"[Error] ct_dir not found: {args.ct_dir}")
        return
    if args.dose_source == "gt" and (not os.path.exists(args.dose_dir)):
        print(f"[Error] dose_dir not found (dose_source=gt): {args.dose_dir}")
        return

    pred_files = list_pred_files(args.pred_dir, args.scan_mode)
    if not pred_files:
        print(f"[Error] No prediction npy files found in: {args.pred_dir} (scan_mode={args.scan_mode})")
        return

    print(f"[Inference] Found {len(pred_files)} pred files (scan_mode={args.scan_mode})")

    print("[Inference] Building indices for fallback matching...")
    ct_index = build_index(args.ct_dir)
    dose_index = build_index(args.dose_dir) if args.dose_source == "gt" else {}

    # Load model
    model = DualHeadResNet3D(in_channels=10).to(device)
    if not os.path.exists(args.model_path):
        print(f"[Error] Model file not found: {args.model_path}")
        return

    try:
        _load_state_dict_safely(model, args.model_path, device)
        print(f"[Inference] Model loaded: {args.model_path}")
    except RuntimeError as e:
        print(f"[Error] Model architecture mismatch or bad checkpoint.\n{e}")
        return

    model.eval()

    img_size = tuple(args.img_size)  # (D,H,W)
    results = []
    missing = []

    with torch.no_grad():
        for pred_name in tqdm(pred_files, desc="Inferencing"):
            try:
                need_gt_dose = (args.dose_source == "gt")
                pred_path, ct_path, gt_dose_path, ct_used, dose_used = resolve_paths_from_pred_name(
                    pred_name=pred_name,
                    pred_dir=args.pred_dir,
                    ct_dir=args.ct_dir,
                    dose_dir=args.dose_dir,
                    ct_index=ct_index,
                    dose_index=dose_index,
                    need_gt_dose=need_gt_dose,
                )

                if pred_path is None or ct_path is None or (need_gt_dose and gt_dose_path is None):
                    raise FileNotFoundError("Matching failed (no corresponding CT/Dose found).")

                x = build_10ch_tensor(
                    pred_path=pred_path,
                    ct_path=ct_path,
                    gt_dose_path=gt_dose_path,
                    img_size=img_size,
                    dose_source=args.dose_source,
                )  # (10,D,H,W)
                x = x.unsqueeze(0).to(device)  # (1,10,D,H,W)

                pred_int, pred_smooth = model(x)

                final_int = round(abs(float(pred_int.item())), 2)
                final_smooth = round(abs(float(pred_smooth.item())), 1)

                pid = extract_patient_id(pred_name) or os.path.splitext(pred_name)[0]
                group = "WL" if "WL" in pid else ("WR" if "WR" in pid else "")

                results.append({
                    "Case": pid,
                    "Group": group,
                    "PredFile": pred_name,
                    "CTFile": ct_used,
                    "DoseFile": dose_used if dose_used is not None else "",
                    "TOW": final_int,
                    "VDP": final_smooth,
                })

            except Exception as e:
                pid = extract_patient_id(pred_name) or os.path.splitext(pred_name)[0]
                missing.append({
                    "Case": pid,
                    "PredFile": pred_name,
                    "error": str(e),
                })
                if args.verbose_missing:
                    print(f"\n[WARN] Skip {pred_name} -> {e}")

    # Save missing matches
    if missing:
        miss_df = pd.DataFrame(missing)
        miss_path = os.path.join(args.output_dir, "missing_matches.csv")
        miss_df.to_csv(miss_path, index=False)
        print(f"[Warn] Missing/failed cases: {len(missing)} -> saved to: {miss_path}")

    # Save full details (전체 로그용)
    df_res = pd.DataFrame(results)
    details_path = os.path.join(args.output_dir, "inference_details.csv")
    df_res.to_csv(details_path, index=False, encoding="utf-8-sig")
    print(f"[Done] Details saved to: {details_path}")

    if len(df_res) == 0:
        print("[Done] No successful inference rows. Check missing_matches.csv")
        return

    # 1 per Case 요약 정보 생성
    df_summary = df_res.groupby("Case", as_index=False).first()
    df_summary = df_summary[["Case", "TOW", "VDP"]]

    # --- 여기서부터 WL/WR 분리 저장 로직 ---
    # Case 컬럼의 마지막 두 글자로 필터링
    df_wl = df_summary[df_summary["Case"].astype(str).str.endswith("WL")].copy()
    df_wr = df_summary[df_summary["Case"].astype(str).str.endswith("WR")].copy()

    # 각각 저장
    wl_path = os.path.join(args.output_dir, f"{model_type}_prediction_WL.csv")
    wr_path = os.path.join(args.output_dir, f"{model_type}_prediction_WR.csv")
    
    df_wl.to_csv(wl_path, index=False, encoding="utf-8-sig")
    df_wr.to_csv(wr_path, index=False, encoding="utf-8-sig")

    # 기존 통합 파일도 필요하다면 유지
    summary_path = os.path.join(args.output_dir, "final_prediction.csv")
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"[Split Save] WL: {len(df_wl)} rows -> {wl_path}")
    print(f"[Split Save] WR: {len(df_wr)} rows -> {wr_path}")
    print(f"[Done] Final prediction saved to: {summary_path}")

# ============================================================
# 5) CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--pred_dir", type=str, required=True, help="Folder containing pred_dose npy files (e.g., predictions_3d)")
    p.add_argument("--ct_dir", type=str, required=True, help="Folder of GT CT_and_Contour npy (e.g., ../dataset/Test/CT_and_Contour)")
    p.add_argument("--dose_dir", type=str, default="", help="Folder of GT Dose npy (e.g., ../dataset/Test/Dose). Needed only if --dose_source gt")

    p.add_argument("--model_path", type=str, required=True, help="Path to best_model.pth")
    p.add_argument("--output_dir", type=str, default="./results", help="Output folder")

    p.add_argument("--img_size", nargs=3, type=int, default=[64, 128, 128], help="(D,H,W) model input size")
    p.add_argument("--dose_source", type=str, default="pred", choices=["pred", "gt"],
                   help="10th channel source: 'pred' uses pred_dose; 'gt' uses GT dose")
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu")

    p.add_argument("--scan_mode", type=str, default="eval_like", choices=["eval_like", "all_npy"],
                   help="How to scan pred_dir: 'eval_like' matches your evaluation filter, 'all_npy' uses all .npy files")

    p.add_argument("--verbose_missing", action="store_true", help="Print each missing/mismatch error")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # dose_source가 pred면 dose_dir 없어도 됨. gt면 필수.
    if args.dose_source == "gt" and (not args.dose_dir):
        raise ValueError("--dose_dir is required when --dose_source gt")
    run_inference(args)