#2025.01.24 unet2D_simam_2_5D
# 새로운 total loss test중이니 나중에 확인 해볼것

import os
import time
from xml.parsers.expat import model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
import gc
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torchvision.ops as ops
from torchviz import make_dot
from torchsummary import summary

from utils.visualization import visualize_prediction, save_gif_from_slices  # ⬅️ 추가된 시각화 모듈
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

from pytorch_msssim import SSIM
from scipy.ndimage import rotate
import random
from collections import defaultdict

# Global parameters
channel_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1]  # 원본 예시대로 8채널 가정
batch_size = 8 # 배치 크기
total_epochs = 1000
check_epoch = 2 # 모델 저장 및 시각화 주기
CHUNK_SIZE = 10 # ✅ 한 번에 메모리에 올릴 훈련 파일 수
slice_window = 5 # 중심 슬라이스를 기준으로 앞뒤로 2장씩(총 5장)
CACHE_SIZE = 100 # LRU 캐시의 최대 크기 

num_workers = 4 # DataLoader의 워커 수
smooth = 1
lamda = 0.5
img_rows = 256
img_cols = 256
clip_min = -300
clip_max = 800
start_index = 0
prescription_dose = prescribed_dose = 26.00
target_value = target_dose = prescribed_dose * 1.1
dvh_loss_fn = None

# ✅ 검증 세트 비율을 20%로 설정 (Train 8 : Val 2)
val_size = 0.2 

# 약 30GB RAM을 사용하고 싶다면, 30,000MB / 30MB/item ≈ 1000개 아이템
# 적절한 값으로 시작하여 메모리 사용량을 모니터링하며 조절
CACHE_SIZE = 100 # LRU 캐시의 최대 크기 

# ------ 2.5D 관련 설정 ------
slice_window = 5 # 중심 슬라이스를 기준으로 앞뒤로 2장씩(총 5장)

data_root = os.environ.get("WBI_DATA_ROOT", "./data")
train_ct_contour_dir = os.path.join(data_root, "Train", "CT_and_Contour")
train_dose_dir = os.path.join(data_root, "Train", "Dose")
# ✅ Test 경로 완전히 삭제됨

save_file = './Model_Weight_2D_SimAM/best_model_weight.pth'
save_check_file = os.path.join(os.environ.get("WBI_OUTPUT_ROOT", "./outputs"), "epoch_image")
grad_cam_name = 'gradcam'


########################################
# make dvh graph function
#########################################
def dvh_curve_cumulative(dose_gy_3d, mask_3d, bins_gy):
    vals = dose_gy_3d[mask_3d > 0]
    if vals.size == 0:
        return None
    v = np.array([(vals >= d).mean() * 100.0 for d in bins_gy], dtype=np.float32)
    return v

def make_qualitative_figure(ct2d, roi_dict_2d, true2d_gy, pred2d_gy, dvh_loss_value, vmax):
    diff = true2d_gy - pred2d_gy

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    ax0, ax1, ax2, ax3 = axes

    ax0.imshow(ct2d, cmap="gray")
    ax0.set_title("Input CT (center slice)")
    ax0.axis("off")

    ax1.imshow(ct2d, cmap="gray")
    for name, m in roi_dict_2d.items():
        if m.sum() > 0:
            ax1.contour(m, levels=[0.5], linewidths=1)
    ax1.set_title("ROI masks (contours)")
    ax1.axis("off")

    im2 = ax2.imshow(true2d_gy, vmin=0, vmax=vmax)
    ax2.set_title("True dose (Gy)")
    ax2.axis("off")
    fig.colorbar(im2, ax=ax2, fraction=0.046)

    im3 = ax3.imshow(pred2d_gy, vmin=0, vmax=vmax)
    ax3.set_title(f"Pred dose (Gy)\nDVH loss={dvh_loss_value:.4f}")
    ax3.axis("off")
    fig.colorbar(im3, ax=ax3, fraction=0.046)

    fig.tight_layout()
    return fig

def make_diff_figure(true2d_gy, pred2d_gy):
    diff = true2d_gy - pred2d_gy
    v = np.max(np.abs(diff)) + 1e-6
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    im = ax.imshow(diff, vmin=-v, vmax=v)
    ax.set_title("Dose diff (True - Pred) [Gy]")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig

def make_dvh_figure(true_gy_3d, pred_gy_3d, roi_vols_3d, bins_gy):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for roi_name, mask3d in roi_vols_3d.items():
        v_true = dvh_curve_cumulative(true_gy_3d, mask3d, bins_gy)
        v_pred = dvh_curve_cumulative(pred_gy_3d, mask3d, bins_gy)
        if v_true is None or v_pred is None:
            continue
        line_true, = ax.plot(bins_gy, v_true, label=f"{roi_name} (true)", linestyle='-')
        ax.plot(bins_gy, v_pred, label=f"{roi_name} (pred)", linestyle='--', color=line_true.get_color())

    if dvh_loss_fn is not None:
        criteria = dvh_loss_fn.clinical
        rx = dvh_loss_fn.rx_gy
        
        crit_ctv_vmax = criteria["CTV_Vmax_max_gy"]
        ax.axvline(x=crit_ctv_vmax, color='red', linestyle=':', linewidth=1.5, label=f'CTV_Vmax_limit ({crit_ctv_vmax:.2f}Gy)')

        ax.plot(7.0, criteria["Heart_V7_max_pct"], 'rx', markersize=8, label='Heart_V7_limit')
        ax.plot(1.5, criteria["Heart_V1p5_max_pct"], 'mx', markersize=8, label='Heart_V1.5_limit')
        ax.plot(8.0, criteria["IpsiLung_V8_max_pct"], 'gx', markersize=8, label='IpsiLung_V8_limit')

        crit_ctv_v95_dose = 0.95 * rx
        crit_ctv_v95_vol = criteria["CTV_V95_min_pct"]
        ax.plot(crit_ctv_v95_dose, crit_ctv_v95_vol, 'bx', markersize=8, label='CTV_V95_goal')

    ax.set_xlabel("Dose (Gy)")
    ax.set_ylabel("Volume (%)")
    ax.set_title("DVH (True: solid, Pred: dashed, Criteria: markers/lines)")
    ax.set_xlim(bins_gy[0], bins_gy[-1])
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


########################################
# Data augmentation functions
########################################
class AvgMeter:
    def __init__(self):
        self.sum = defaultdict(float)
        self.cnt = defaultdict(int)

    def update(self, key, value):
        if hasattr(value, "detach"):
            value = value.detach().float().mean().item()
        self.sum[key] += float(value)
        self.cnt[key] += 1

    def avg(self, key, default=0.0):
        c = self.cnt.get(key, 0)
        return self.sum[key] / c if c > 0 else default


def get_augment_params():
    do_flip = random.random() < 0.5
    if random.random() < 0.5:
        angles = (-10, -8, -5, -2, 2, 5, 8, 10)
        angle = random.choice(angles)
    else:
        angle = 0
    return do_flip, angle

def apply_transforms(image, do_flip, angle):
    if angle != 0:
        if image.ndim == 2: 
             image = rotate(image, angle, reshape=False, order=1, mode='nearest')
        else: 
            image_rotated = np.zeros_like(image)
            image_rotated[:, :, 0] = rotate(image[:, :, 0], angle, reshape=False, order=1, mode='nearest')
            for c in range(1, image.shape[2]):
                rotated_mask = rotate(image[:, :, c], angle, reshape=False, order=0, mode='nearest')
                image_rotated[:, :, c] = (rotated_mask > 0.5).astype(image.dtype)
            image = image_rotated
    return image.copy()

def _apply_ct_intensity_aug_2d(image, p_affine=0.5, p_noise=0.5):
    image_to_aug = image.copy()
    ct_slice = image_to_aug[:, :, 0]
    
    if random.random() < p_affine:
        a = random.uniform(0.9, 1.1)
        b = random.uniform(-50, 50)
        ct_slice = a * ct_slice + b
    if random.random() < p_noise:
        sigma = random.uniform(3.0, 8.0)
        ct_slice = ct_slice + np.random.normal(0, sigma, ct_slice.shape)
    
    image_to_aug[:, :, 0] = np.clip(ct_slice, clip_min, clip_max)
    return image_to_aug

########################################
# Loss functions & Metrics
########################################

def _safe_div(a, b, eps=1e-6):
    return a / (b + eps)

def soft_vx_percent(dose_gy, mask01, thr_gy, tau=0.25):
    p = torch.sigmoid((dose_gy - thr_gy) / tau)
    frac = _safe_div((p * mask01).sum(dim=(1,2,3)), mask01.sum(dim=(1,2,3)))
    return frac * 100.0

def mean_dose_gy(dose_gy, mask01):
    s = (dose_gy * mask01).sum(dim=(1,2,3))
    v = mask01.sum(dim=(1,2,3))
    return _safe_div(s, v)

def soft_dmax_gy(dose_gy, mask01, tau_max=0.5):
    B = dose_gy.shape[0]
    x = dose_gy.view(B, -1)
    m = mask01.view(B, -1)
    neg_inf = torch.finfo(x.dtype).min
    x_masked = torch.where(m > 0.5, x, torch.tensor(neg_inf, device=x.device, dtype=x.dtype))
    return tau_max * torch.logsumexp(x_masked / tau_max, dim=1)

class DVHConstraintLoss2D(nn.Module):
    def __init__(
        self,
        rx_gy=26.0,
        target_factor=1.1,
        supersample_factor=1,
        tau=0.25,
        tau_max=0.5,
        gate_by_true=True,
        strict=None,
        clinical=None,
        weights=None,
        gt_offsets=None, 
        v95_offset_mode="mul",
        v95_offset_value=1.02,
    ):
        super().__init__()
        self.rx_gy = float(rx_gy)
        self.target_factor = float(target_factor)
        self.supersample_factor = int(supersample_factor)
        self.tau = float(tau)
        self.tau_max = float(tau_max)
        self.gate_by_true = bool(gate_by_true)
        self.v95_offset_mode = v95_offset_mode
        self.v95_offset_value = v95_offset_value

        if clinical is None:
            clinical = {
                "CTV_V95_min_pct": 95.0, "CTV_Vmax_max_gy": 1.07 * self.rx_gy,
                "Heart_V7_max_pct": 3.0, "Heart_V1p5_max_pct": 30.0,
                "IpsiLung_V8_max_pct": 15.0, "ContraLung_Dmean_max_gy": 2.0,
                "ContraBreast_Dmean_max_gy": 3.0,
            }
        if strict is None:
            strict = {
                "CTV_V95_min_pct": 97.0, "CTV_Vmax_max_gy": 1.05 * self.rx_gy,
                "Heart_V7_max_pct": 2.5, "Heart_V1p5_max_pct": 25.0,
                "IpsiLung_V8_max_pct": 13.0, "ContraLung_Dmean_max_gy": 1.8,
                "ContraBreast_Dmean_max_gy": 2.8,
            }
        if weights is None:
            weights = {
                "CTV_V95": 20.0, "CTV_Vmax": 20.0, "Heart_V7": 2.0, "Heart_V1p5": 2.0,
                "IpsiLung_V8": 2.0, "ContraLung_Dmean": 1.5, "ContraBreast_Dmean": 1.5,
            }

        if gt_offsets is None:
            self.gt_offsets = {
                "CTV_V95_pct": 1.0,  
                "CTV_Vmax_gy": 0.0,  
                "Heart_V7_pct": 0.5,
                "Heart_V1p5_pct": 1.0,
                "IpsiLung_V8_pct": 1.0,
                "ContraLung_Dmean_gy": 0.2,
                "ContraBreast_Dmean_gy": 0.2,
            }
        else:
            self.gt_offsets = gt_offsets

        self.clinical = clinical
        self.strict = strict
        self.weights = weights

    def _to_gy(self, y_norm):
        return y_norm * (self.rx_gy * self.target_factor)

    def _apply_v95_offset(self, v95_pct: torch.Tensor) -> torch.Tensor:
        if self.v95_offset_mode is None:
            return v95_pct
        if self.v95_offset_mode == "mul":
            return v95_pct * float(self.v95_offset_value)
        if self.v95_offset_mode == "add":
            return v95_pct + float(self.v95_offset_value)
        raise ValueError(f"Unknown v95_offset_mode: {self.v95_offset_mode}")

    def _upsample(self, dose_gy, masks):
        if self.supersample_factor <= 1:
            return dose_gy, masks
        sf = self.supersample_factor
        dose_up = F.interpolate(dose_gy, scale_factor=sf, mode="bilinear", align_corners=False)
        masks_up = {
            k: F.interpolate(m.float(), scale_factor=sf, mode="bilinear", align_corners=False)
            for k, m in masks.items()
        }
        return dose_up, masks_up

    def forward(self, y_pred_norm, y_true_norm, roi_masks, meter=None):
        pred_gy = self._to_gy(y_pred_norm)
        true_gy = self._to_gy(y_true_norm)

        pred_gy, roi_masks = self._upsample(pred_gy, roi_masks)
        true_gy, _ = self._upsample(true_gy, roi_masks)

        roi_masks = {k: (v >= 0.5).float() for k, v in roi_masks.items()}

        loss = torch.zeros([], device=pred_gy.device)

        # CTV
        if "CTV" in roi_masks and roi_masks["CTV"].sum() > 0:
            m = roi_masks["CTV"]
            thr_v95 = 0.95 * self.rx_gy
            pred_v95 = soft_vx_percent(pred_gy, m, thr_v95, self.tau)
            true_v95 = soft_vx_percent(true_gy, m, thr_v95, self.tau)
            pred_v95_adj = self._apply_v95_offset(pred_v95)
            true_v95_adj = self._apply_v95_offset(true_v95)

            if (not self.gate_by_true) or torch.all(true_v95_adj >= self.clinical["CTV_V95_min_pct"] - 0.5):
                p_strict = F.relu(self.strict["CTV_V95_min_pct"] - pred_v95_adj)
                p_gt = F.relu((true_v95_adj - self.gt_offsets["CTV_V95_pct"]) - pred_v95_adj)
                final_p = torch.max(p_strict, p_gt)

                if meter is not None:
                    meter.update("DVH_CTV_V95", final_p)
                loss = loss + self.weights["CTV_V95"] * final_p.mean()

            pred_vmax = soft_dmax_gy(pred_gy, m, self.tau_max)
            true_vmax = soft_dmax_gy(true_gy, m, self.tau_max)

            if (not self.gate_by_true) or torch.all(true_vmax <= self.clinical["CTV_Vmax_max_gy"] + 0.05):
                p_strict = F.relu(pred_vmax - self.strict["CTV_Vmax_max_gy"])
                p_gt = F.relu(pred_vmax - (true_vmax + self.gt_offsets["CTV_Vmax_gy"]))
                final_p = torch.max(p_strict, p_gt)

                if meter is not None:
                    meter.update("DVH_CTV_Vmax", final_p)
                loss = loss + self.weights["CTV_Vmax"] * final_p.mean()

        # Heart
        if "Heart" in roi_masks and roi_masks["Heart"].sum() > 0:
            m = roi_masks["Heart"]
            pred_v7 = soft_vx_percent(pred_gy, m, 7.0, self.tau)
            true_v7 = soft_vx_percent(true_gy, m, 7.0, self.tau)

            if (not self.gate_by_true) or torch.all(true_v7 <= self.clinical["Heart_V7_max_pct"] + 0.5):
                p_strict = F.relu(pred_v7 - self.strict["Heart_V7_max_pct"])
                p_gt = F.relu(pred_v7 - (true_v7 + self.gt_offsets["Heart_V7_pct"]))
                final_p = torch.max(p_strict, p_gt)

                if meter is not None:
                    meter.update("DVH_Heart_V7", final_p)
                loss = loss + self.weights["Heart_V7"] * final_p.mean()

            pred_v15 = soft_vx_percent(pred_gy, m, 1.5, self.tau)
            true_v15 = soft_vx_percent(true_gy, m, 1.5, self.tau)

            if (not self.gate_by_true) or torch.all(true_v15 <= self.clinical["Heart_V1p5_max_pct"] + 1.0):
                p_strict = F.relu(pred_v15 - self.strict["Heart_V1p5_max_pct"])
                p_gt = F.relu(pred_v15 - (true_v15 + self.gt_offsets["Heart_V1p5_pct"]))
                final_p = torch.max(p_strict, p_gt)

                if meter is not None:
                    meter.update("DVH_Heart_V1p5", final_p)
                loss = loss + self.weights["Heart_V1p5"] * final_p.mean()

        # Ipsi Lung
        if "Ipsi_Lung" in roi_masks and roi_masks["Ipsi_Lung"].sum() > 0:
            m = roi_masks["Ipsi_Lung"]
            pred_v8 = soft_vx_percent(pred_gy, m, 8.0, self.tau)
            true_v8 = soft_vx_percent(true_gy, m, 8.0, self.tau)

            if (not self.gate_by_true) or torch.all(true_v8 <= self.clinical["IpsiLung_V8_max_pct"] + 1.0):
                p_strict = F.relu(pred_v8 - self.strict["IpsiLung_V8_max_pct"])
                p_gt = F.relu(pred_v8 - (true_v8 + self.gt_offsets["IpsiLung_V8_pct"]))
                final_p = torch.max(p_strict, p_gt)

                if meter is not None:
                    meter.update("DVH_IpsiLung_V8", final_p)
                loss = loss + self.weights["IpsiLung_V8"] * final_p.mean()

        # Contra Lung Dmean
        if "Contra_Lung" in roi_masks and roi_masks["Contra_Lung"].sum() > 0:
            m = roi_masks["Contra_Lung"]
            pred_dm = mean_dose_gy(pred_gy, m)
            true_dm = mean_dose_gy(true_gy, m)

            if (not self.gate_by_true) or torch.all(true_dm <= self.clinical["ContraLung_Dmean_max_gy"] + 0.1):
                p_strict = F.relu(pred_dm - self.strict["ContraLung_Dmean_max_gy"])
                p_gt = F.relu(pred_dm - (true_dm + self.gt_offsets["ContraLung_Dmean_gy"]))
                final_p = torch.max(p_strict, p_gt)

                if meter is not None:
                    meter.update("DVH_ContraLung_Dmean", final_p)
                loss = loss + self.weights["ContraLung_Dmean"] * final_p.mean()

        # Contra Breast Dmean
        if "Contra_Breast" in roi_masks and roi_masks["Contra_Breast"].sum() > 0:
            m = roi_masks["Contra_Breast"]
            pred_dm = mean_dose_gy(pred_gy, m)
            true_dm = mean_dose_gy(true_gy, m)

            if (not self.gate_by_true) or torch.all(true_dm <= self.clinical["ContraBreast_Dmean_max_gy"] + 0.1):
                p_strict = F.relu(pred_dm - self.strict["ContraBreast_Dmean_max_gy"])
                p_gt = F.relu(pred_dm - (true_dm + self.gt_offsets["ContraBreast_Dmean_gy"]))
                final_p = torch.max(p_strict, p_gt)

                if meter is not None:
                    meter.update("DVH_ContraBreast_Dmean", final_p)
                loss = loss + self.weights["ContraBreast_Dmean"] * final_p.mean()

        return loss

def dice_coef(y_true, y_pred):
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)

class SSIM3DLoss(nn.Module):
    def __init__(self, data_range=1.0, channel=1):
        super(SSIM3DLoss, self).__init__()
        self.ssim_loss = SSIM(data_range=data_range, size_average=True, channel=channel, spatial_dims=2)

    def forward(self, x, y):
        return 1 - self.ssim_loss(x, y)

ssim_loss_fn = SSIM3DLoss(data_range=prescription_dose * 1.1, channel=1) 

def total_loss(y_true, y_pred, roi_masks, meter=None):
    y_pred_gy = y_pred * target_dose
    y_true_gy = y_true * target_dose

    alpha = 0.85
    l1_loss = F.l1_loss(y_pred_gy, y_true_gy)
    ssim_loss = ssim_loss_fn(y_pred_gy, y_true_gy)
    base_loss = (alpha * l1_loss) + ((1 - alpha) * ssim_loss)

    dvh_penalty = dvh_loss_fn(y_pred, y_true, roi_masks, meter=meter)

    base_out = base_loss * 3.5
    dvh_out = dvh_penalty * 6.5 

    #total = base_out + dvh_out
    # test with out dvh loss
    total = base_out

    if meter is not None:
        meter.update("BaseLoss_scaled", base_out)
        meter.update("DVH_scaled", dvh_out)
        meter.update("TotalLoss", total)

    return total


#######################################
# SimAM Layer & U-Net
#######################################
class SimAM(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(SimAM, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        mean = torch.mean(x, dim=[2, 3], keepdim=True)
        var = torch.mean((x - mean)**2, dim=[2, 3], keepdim=True)
        norm_x = (x - mean) / torch.sqrt(var + self.epsilon)
        return x * torch.sigmoid(norm_x)

def conv_block(in_feat, out_feat):
    block = nn.Sequential(
        nn.Conv2d(in_feat, out_feat, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_feat),
        SimAM()
    )
    return block

def up_block(in_feat, out_feat):
    up = nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)
    return up

class UNet2D_SimAM(nn.Module):
    def __init__(self, in_channels=9, out_channels=1):
        super(UNet2D_SimAM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = conv_block(128, 128)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = conv_block(128, 256)

        self.up6_convT = up_block(256, 128)
        self.up6_block = conv_block(256, 128)
        self.up7_convT = up_block(128, 128)
        self.up7_block = conv_block(256, 128)
        self.up8_convT = up_block(128, 64)
        self.up8_block = conv_block(128, 64)
        self.up9_convT = up_block(64, 32)
        self.up9_block = conv_block(64, 32)

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c1 = self.conv1(x)  
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)  
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)  
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)  
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)  

        x6 = self.up6_convT(c5)  
        x6 = torch.cat([x6, c4], dim=1)  
        x6 = self.up6_block(x6)  

        x7 = self.up7_convT(x6)  
        x7 = torch.cat([x7, c3], dim=1)  
        x7 = self.up7_block(x7)  

        x8 = self.up8_convT(x7)  
        x8 = torch.cat([x8, c2], dim=1)  
        x8 = self.up8_block(x8)  

        x9 = self.up9_convT(x8)  
        x9 = torch.cat([x9, c1], dim=1)  
        x9 = self.up9_block(x9)  

        out = self.final_conv(x9)  
        out = self.sigmoid(out)
        return out

#######################################
# CustomDataset
#######################################
class CustomDataset(Dataset):
    def __init__(self, ct_contour_paths, dose_paths, slice_window, channel_weights, max_cache_size=CACHE_SIZE, augment=False):
        self.ct_contour_paths = ct_contour_paths
        self.dose_paths = dose_paths
        self.slice_window = slice_window
        self.channel_weights = channel_weights
        self.augment = augment 
        
        self.clip_min = clip_min
        self.clip_max = clip_max
        
        self.file_cache = {}
        self.slice_cache = {}
        self.max_cache_size = max_cache_size
        self.cache_keys = []
        
        self.indices = []
        for file_idx, ct_path in enumerate(self.ct_contour_paths):
            try:
                with open(ct_path, 'rb') as f:
                    version = np.lib.format.read_magic(f)
                    shape, _, _ = np.lib.format._read_array_header(f, version)
                    Z = shape[2]
                
                for z in range(self.slice_window, Z - self.slice_window):
                    self.indices.append((file_idx, z))
            except Exception as e:
                print(f"Warning: Could not read header from {ct_path}. Skipping file. Error: {e}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, z = self.indices[idx] 
        
        use_cache = not self.augment 
        cache_key = (file_idx, z) 
        
        if use_cache and (cache_key in self.slice_cache):
            return self.slice_cache[cache_key]

        if file_idx not in self.file_cache:
            ct_contour_data = np.load(self.ct_contour_paths[file_idx], mmap_mode='r')
            dose_data = np.load(self.dose_paths[file_idx], mmap_mode='r')
            self.file_cache[file_idx] = (ct_contour_data, dose_data)
        else:
            ct_contour_data, dose_data = self.file_cache[file_idx]

        do_flip = False
        angle = 0
        if self.augment:
            do_flip, angle = get_augment_params()

        y = dose_data[:, :, z].copy()
        if self.augment:
            y = apply_transforms(y, do_flip, angle)

        slices = []
        H, W, _, C = ct_contour_data.shape
        
        for offset in range(-self.slice_window, self.slice_window + 1):
            slice_data = ct_contour_data[:, :, z + offset, :].copy() 
            
            if self.augment:
                slice_data = apply_transforms(slice_data, do_flip, angle)
                slice_data = _apply_ct_intensity_aug_2d(slice_data)

            slice_data = np.clip(slice_data, self.clip_min, self.clip_max)
            for c_idx in range(C):
                slice_data[:, :, c_idx] *= self.channel_weights[c_idx]
            slices.append(slice_data)

        x = np.concatenate(slices, axis=-1)
        x = torch.from_numpy(np.transpose(x, (2, 0, 1)).astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32)[None, ...])

        per_slice_C = 9
        center_base = self.slice_window * per_slice_C 

        roi_masks = {
            "A_LAD": x[center_base+1:center_base+2].clone(),
            "CTV": x[center_base+2:center_base+3].clone(),
            "Contra_Breast": x[center_base+3:center_base+4].clone(),
            "External": x[center_base+4:center_base+5].clone(),
            "Heart": x[center_base+5:center_base+6].clone(),
            "Ipsi_Lung": x[center_base+6:center_base+7].clone(),
            "Contra_Lung": x[center_base+7:center_base+8].clone(),
            "RING": x[center_base+8:center_base+9].clone(),
        }
        
        result = (x, y, roi_masks)
        
        if use_cache:
            if len(self.slice_cache) >= self.max_cache_size:
                oldest_key = self.cache_keys.pop(0)
                if oldest_key in self.slice_cache:
                     del self.slice_cache[oldest_key]
            self.slice_cache[cache_key] = result
            self.cache_keys.append(cache_key)
        
        return result


def prepare_single_test_case(ct_contour_path, dose_path, slice_window, channel_weights):
    print(f"Preparing single test case for visualization from: {os.path.basename(ct_contour_path)}")

    ct_contour_data = np.load(ct_contour_path) 
    dose_data = np.load(dose_path)             

    ct_contour_data = np.clip(ct_contour_data, clip_min, clip_max)
    dose_data = np.clip(dose_data, clip_min, clip_max)

    H, W, Z, C = ct_contour_data.shape

    roi_vols = {
        "CTV": (ct_contour_data[:, :, :, 2].transpose(2,0,1) > 0.5),
        "Heart": (ct_contour_data[:, :, :, 5].transpose(2,0,1) > 0.5),
        "Ipsi_Lung": (ct_contour_data[:, :, :, 6].transpose(2,0,1) > 0.5),
        "Contra_Lung": (ct_contour_data[:, :, :, 7].transpose(2,0,1) > 0.5),
        "Contra_Breast": (ct_contour_data[:, :, :, 3].transpose(2,0,1) > 0.5),
    }

    inputs_list = []
    labels_list = []

    for z in range(slice_window, Z - slice_window):
        slices_ = []
        for offset in range(-slice_window, slice_window + 1):
            slice_data = ct_contour_data[:, :, z + offset, :]
            for c_idx in range(C):
                slice_data[:, :, c_idx] *= channel_weights[c_idx]
            slices_.append(slice_data)

        x = np.concatenate(slices_, axis=-1) 
        inputs_list.append(np.transpose(x, (2, 0, 1)).astype(np.float32))

        y = dose_data[:, :, z]
        labels_list.append(y.astype(np.float32)[None, ...])

    return np.array(inputs_list), np.array(labels_list), roi_vols


# ✅ 파라미터에서 test 파일 경로 삭제
def train_model_on_chunk(model, optimizer,
                         train_ct_contour_paths, train_dose_paths,
                         total_epochs, batch_size, save_file,
                         slice_window, channel_weights, min_delta=0.001, patience=30, writer=None, CACHE_SIZE=100):

    labels = ['WR' if 'WR' in f else 'WL' for f in train_ct_contour_paths]
    
    # ✅ val_size (0.2) 적용하여 8:2로 나눔
    train_ct_paths, val_ct_paths, train_dose_paths, val_dose_paths = train_test_split(
        train_ct_contour_paths,
        train_dose_paths,
        test_size=val_size, 
        random_state=42,
        stratify=labels
    )

    val_ct_paths_wl = [p for p in val_ct_paths if 'WL' in p]
    val_dose_paths_wl = [p for p in val_dose_paths if 'WL' in os.path.basename(p)]
    val_ct_paths_wr = [p for p in val_ct_paths if 'WR' in p]
    val_dose_paths_wr = [p for p in val_dose_paths if 'WR' in os.path.basename(p)]

    print("\n--- Stratified Data Split ---")
    print(f"Total Training set: {len(train_ct_paths)} files (will be chunked per epoch)")
    print(f"Validation set (Total): {len(val_ct_paths)} files")
    print(f"  - WR cases for validation: {len(val_ct_paths_wr)}")
    print(f"  - WL cases for validation: {len(val_ct_paths_wl)}")
    print("-----------------------------\n")

    val_dataset = CustomDataset(val_ct_paths, val_dose_paths, slice_window, channel_weights, max_cache_size=CACHE_SIZE, augment=False)
    val_dataset_wl = CustomDataset(val_ct_paths_wl, val_dose_paths_wl, slice_window, channel_weights, max_cache_size=CACHE_SIZE, augment=False)
    val_dataset_wr = CustomDataset(val_ct_paths_wr, val_dose_paths_wr, slice_window, channel_weights, max_cache_size=CACHE_SIZE, augment=False)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_loader_wl = DataLoader(val_dataset_wl, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_dataset_wl else None
    val_loader_wr = DataLoader(val_dataset_wr, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_dataset_wr else None

    # ✅ 시각화를 Test 데이터 대신 Validation 세트 내에서 수행하도록 변경
    vis_cases = {}
    if val_ct_paths:
        wl_idx = next((i for i, p in enumerate(val_ct_paths) if 'WL' in os.path.basename(p)), -1)
        if wl_idx != -1:
            print("\n--- Preparing WL visualization case from Validation set ---")
            vis_cases['WL'] = {'data': prepare_single_test_case(val_ct_paths[wl_idx], val_dose_paths[wl_idx], slice_window, channel_weights)}
        wr_idx = next((i for i, p in enumerate(val_ct_paths) if 'WR' in os.path.basename(p)), -1)
        if wr_idx != -1:
            print("\n--- Preparing WR visualization case from Validation set ---")
            vis_cases['WR'] = {'data': prepare_single_test_case(val_ct_paths[wr_idx], val_dose_paths[wr_idx], slice_window, channel_weights)}
    if not vis_cases:
        print("Warning: No suitable WL/WR validation files found for visualization.")

    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop_triggered = False

    for epoch in tqdm(range(1, total_epochs + 1)):
        model.train()
        train_meter = AvgMeter()

        shuffled_indices = shuffle(range(len(train_ct_paths)), random_state=epoch)
        epoch_train_ct_paths = [train_ct_paths[i] for i in shuffled_indices]
        epoch_train_dose_paths = [train_dose_paths[i] for i in shuffled_indices]

        ct_chunks = [epoch_train_ct_paths[i:i + CHUNK_SIZE] for i in range(0, len(epoch_train_ct_paths), CHUNK_SIZE)]
        dose_chunks = [epoch_train_dose_paths[i:i + CHUNK_SIZE] for i in range(0, len(epoch_train_dose_paths), CHUNK_SIZE)]

        print(f"\nEpoch {epoch}: Training on {len(train_ct_paths)} files in {len(ct_chunks)} chunks.")

        for i, (ct_chunk, dose_chunk) in enumerate(zip(ct_chunks, dose_chunks)):
            train_dataset = CustomDataset(ct_chunk, dose_chunk, slice_window, channel_weights, max_cache_size=CACHE_SIZE, augment=True)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

            for x, y, masks in tqdm(train_loader, desc=f"Epoch {epoch}, Chunk {i+1}/{len(ct_chunks)}", leave=False):
                x, y = x.to(device), y.to(device)
                for k in masks: masks[k] = masks[k].to(device)

                optimizer.zero_grad()
                y_pred = model(x)
                loss = total_loss(y, y_pred, masks, meter=train_meter)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_meter.update("Dice", dice_coef(y.cpu(), y_pred.cpu()))
            
            del train_dataset, train_loader
            gc.collect()

        model.eval()
        val_meter = AvgMeter()
        with torch.no_grad():
            for x_val, y_val, masks_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                for k in masks_val: masks_val[k] = masks_val[k].to(device)
                y_pred_val = model(x_val)
                total_loss(y_val, y_pred_val, masks_val, meter=val_meter)
                val_meter.update("Dice", dice_coef(y_val.cpu(), y_pred_val.cpu()))

        val_loss_avg = val_meter.avg('TotalLoss')

        def evaluate_subgroup(loader):
            if not loader: return 0.0, 0.0
            meter = AvgMeter()
            with torch.no_grad():
                for x, y, m in loader:
                    x, y = x.to(device), y.to(device)
                    for k in m: m[k] = m[k].to(device)
                    y_pred = model(x)
                    total_loss(y, y_pred, m, meter=meter)
                    meter.update("Dice", dice_coef(y.cpu(), y_pred.cpu()))
            return meter.avg('TotalLoss'), meter.avg('Dice')

        val_loss_wl, val_dice_wl = evaluate_subgroup(val_loader_wl)
        val_loss_wr, val_dice_wr = evaluate_subgroup(val_loader_wr)

        if writer:
            writer.add_scalar('Loss/Train', train_meter.avg('TotalLoss'), epoch)
            writer.add_scalar('Dice/Train', train_meter.avg('Dice'), epoch)
            writer.add_scalar('Loss/Val_Total', val_loss_avg, epoch)
            writer.add_scalar('Dice/Val_Total', val_meter.avg('Dice'), epoch)
            writer.add_scalar('Loss/Val_WL', val_loss_wl, epoch)
            writer.add_scalar('Dice/Val_WL', val_dice_wl, epoch)
            writer.add_scalar('Loss/Val_WR', val_loss_wr, epoch)
            writer.add_scalar('Dice/Val_WR', val_dice_wr, epoch)

        print(
            f"\nEpoch [{epoch}/{total_epochs}] - Loss: {train_meter.avg('TotalLoss'):.4f}, Val Loss: {val_loss_avg:.4f} "
            f"| Val(WL): {val_loss_wl:.4f}, Val(WR): {val_loss_wr:.4f}"
        )

        dvh_print_keys = {
            "CTV_V95": "DVH_CTV_V95", "CTV_Vmax": "DVH_CTV_Vmax", "Heart_V7": "DVH_Heart_V7",
            "Heart_V1p5": "DVH_Heart_V1p5", "IpsiLung_V8": "DVH_IpsiLung_V8",
            "ContraLung_Dm": "DVH_ContraLung_Dmean", "ContraBreast_Dm": "DVH_ContraBreast_Dmean"
        }
        
        val_dvh_str = " | ".join([f"{name}: {val_meter.avg(key):.4f}" for name, key in dvh_print_keys.items() if val_meter.cnt.get(key, 0) > 0])
        if val_dvh_str:
            print(f"  Avg Val DVH Penalties: {val_dvh_str}")

        if writer:
            for name, key in dvh_print_keys.items():
                if val_meter.cnt.get(key, 0) > 0:
                    writer.add_scalar(f'DVH_Val/{name}_penalty', val_meter.avg(key), epoch)

        if val_loss_avg < best_val_loss - min_delta:
            best_val_loss = val_loss_avg
            epochs_no_improve = 0
            save_dir = os.path.dirname(save_file)
            best_model_path = os.path.join(save_dir, "best_model_weight.pth")
            print(f"Validation loss improved to {best_val_loss:.4f}. Saving best model to {best_model_path}")
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epochs.")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}!")
                early_stop_triggered = True

        if epoch % check_epoch == 0 and not early_stop_triggered:
            print(f"--- Saving checkpoint for epoch {epoch} ---")
            save_dir = os.path.dirname(save_file)
            checkpoint_path = os.path.join(save_dir, f"{epoch}_model_weight.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

            if vis_cases:
                for case_type, case_info in vis_cases.items():
                    print(f"--- Generating visualization for {case_type} case ---")
                    vis_inputs_np, vis_labels_np, vis_roi_vols = case_info['data']
                    C0 = len(channel_weights)
                    center = C0 * slice_window
                    model.eval()
                    vis_inputs_tensor = torch.from_numpy(vis_inputs_np)
                    pred_chunks = []
                    with torch.no_grad():
                        for i in range(0, len(vis_inputs_tensor), batch_size):
                            bx = vis_inputs_tensor[i:i + batch_size].to(device, non_blocking=True)
                            pred = model(bx)
                            pred_chunks.append(pred.detach().cpu().numpy())
                        torch.cuda.empty_cache()
                    pred_vol = np.concatenate(pred_chunks, axis=0).squeeze(1)
                    true_vol = vis_labels_np.squeeze(1)
                    all_trues_gy = true_vol * target_dose
                    all_preds_gy = pred_vol * target_dose
                    if writer is not None:
                        Z_valid = all_trues_gy.shape[0]
                        mid = Z_valid // 2
                        t = all_trues_gy[mid]
                        p = all_preds_gy[mid]
                        ct2d = vis_inputs_np[mid, center + 0]
                        roi_dict_2d = {
                            "CTV": (vis_inputs_np[mid, center + 2] > 0.5).astype(np.uint8),
                            "Heart": (vis_inputs_np[mid, center + 5] > 0.5).astype(np.uint8),
                        }
                        fig_qual = make_qualitative_figure(ct2d, roi_dict_2d, t, p, 0, vmax=target_dose)
                        writer.add_figure(f"Vis/{case_type}_qualitative", fig_qual, global_step=epoch)
                        fig_qual.savefig(os.path.join(save_check_file, f"epoch_{epoch}_{case_type}_qualitative.png"))
                        plt.close(fig_qual)
                        
                        cropped_roi_vols = {}
                        for roi_name, mask_vol in vis_roi_vols.items():
                            cropped_roi_vols[roi_name] = mask_vol[slice_window:-slice_window, :, :]
                        
                        fig_dvh = make_dvh_figure(all_trues_gy, all_preds_gy, cropped_roi_vols, np.linspace(0, target_dose, 200))
                        writer.add_figure(f"Vis/{case_type}_DVH", fig_dvh, global_step=epoch)
                        fig_dvh.savefig(os.path.join(save_check_file, f"epoch_{epoch}_{case_type}_dvh.png"))
                        plt.close(fig_dvh)
                model.train()
        
        if early_stop_triggered:
            print("Exiting training loop due to early stopping.")
            break

    print("Training process finished.")

def main():
    global dvh_loss_fn
    dvh_loss_fn = DVHConstraintLoss2D(
        rx_gy=prescription_dose,
        target_factor=1.1,
        supersample_factor=1, 
        tau=0.25,
        tau_max=0.5,
        gate_by_true=True,
        v95_offset_mode="mul",
        v95_offset_value=1.02,
    )

    save_dir = os.path.dirname(save_file)
    os.makedirs(save_dir, exist_ok=True)

    in_channels = (2 * slice_window + 1) * len(channel_weights)
    model = UNet2D_SimAM(in_channels=in_channels, out_channels=1).to(device)

    x = torch.randn(1, in_channels, 256, 256).to(device)
    y = model(x)

    summary(model, (in_channels, 256, 256))
    graph = make_dot(y, params=dict(model.named_parameters()))
    graph.render("Simam_model", format='png')

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'bn' in name or 'norm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.Adam([
        {'params': decay_params, 'weight_decay': 1e-6},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=1e-5)

    latest_epoch = -1
    latest_model_path = None
    if os.path.isdir(save_dir):
        for f in os.listdir(save_dir):
            if f.endswith('_model_weight.pth'):
                try:
                    epoch_num = int(f.split('_')[0])
                    if epoch_num > latest_epoch:
                        latest_epoch = epoch_num
                        latest_model_path = os.path.join(save_dir, f)
                except (ValueError, IndexError):
                    continue 

    if latest_model_path:
        print(f"Loading latest model from epoch {latest_epoch}: {latest_model_path}")
        model.load_state_dict(torch.load(latest_model_path, map_location=device))
    elif os.path.exists(save_file): 
        print(f"No epoch-based model found. Loading from default path: {save_file}")
        model.load_state_dict(torch.load(save_file, map_location=device))
    else:
        print("Creating a new model...")

    train_files = sorted([os.path.join(train_ct_contour_dir, f) 
                          for f in os.listdir(train_ct_contour_dir) if f.endswith('.npy')])
    train_dose_files = sorted([os.path.join(train_dose_dir, f) 
                               for f in os.listdir(train_dose_dir) if f.endswith('.npy')])

    # ✅ test_files, test_dose_files 로드 로직 완전히 삭제

    log_dir = os.path.join("runs", f"SimAM_2p5D_{time.strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_dir=log_dir)
    print("TensorBoard logdir:", log_dir)

    # ✅ train_model_on_chunk 파라미터에서 test 파일 부분 삭제됨
    train_model_on_chunk(
        model, optimizer, 
        train_files, train_dose_files,
        total_epochs, batch_size, save_file,
        slice_window, channel_weights, writer=writer,
    )

    torch.cuda.empty_cache()
    gc.collect()

    print("Training done!")


def creatFolder(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print('Error : Creating directory. ' + path + OSError)

if __name__ == '__main__':
    creatFolder(save_check_file)
    print("Start training SimAM with 2.5D input in PyTorch")
    main()
    print("Training done!")
