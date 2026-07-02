import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import gc
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import SSIM
import torchvision.transforms.functional as TF
import random
from collections import defaultdict
# from torchsummary import summary  # Removed due to incompatibility with list inputs

# ------------------------------
#   Model blocks (from Hd-unet.txt)
# ------------------------------
class SimAM(nn.Module):
    def __init__(self, eps=1e-4):
        super(SimAM, self).__init__()
        self.eps = eps

    def forward(self, x):
        b, c, d, h, w = x.size()
        n = d * h * w - 1
        d_mu = x - x.mean(dim=[2, 3, 4], keepdim=True)
        d_sigma = d_mu.pow(2).sum(dim=[2, 3, 4], keepdim=True) / n
        y = d_mu.pow(2) / (4 * (d_sigma + self.eps)) + 0.5
        return x * torch.sigmoid(y)


class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super(SingleConv, self).__init__()
        self.single_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=True),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            SimAM()
        )
    def forward(self, x):
        return self.single_conv(x)


class DenseConvolve(nn.Module):
    def __init__(self, in_ch, growth_rate=16, stride=(1, 1, 1)):
        super(DenseConvolve, self).__init__()
        self.single_conv = nn.Sequential(
            nn.Conv3d(in_ch, growth_rate, kernel_size=(3, 3, 3), padding=1, stride=stride, bias=True),
            nn.InstanceNorm3d(growth_rate, affine=True),
            nn.ReLU(inplace=True),
            SimAM()
        )
    def forward(self, x):
        return torch.cat((self.single_conv(x), x), dim=1)


class DenseDownsample(nn.Module):
    def __init__(self, in_ch, growth_rate=16, stride=(2, 2, 2)):
        super(DenseDownsample, self).__init__()
        self.single_conv = nn.Sequential(
            nn.Conv3d(in_ch, growth_rate, kernel_size=(3, 3, 3), padding=1, stride=stride, bias=True),
            nn.InstanceNorm3d(growth_rate, affine=True),
            nn.ReLU(inplace=True),
            SimAM()
        )
        self.pooling = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
    def forward(self, x):
        return torch.cat((self.single_conv(x), self.pooling(x)), dim=1)


class UNetUpsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNetUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(3, 3, 3), padding=1, stride=(1, 1, 1), bias=True),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_ch, growth_rate=16):
        super(Encoder, self).__init__()
        self.encoder_1 = nn.Sequential(
            DenseConvolve(in_ch, growth_rate),
            DenseConvolve(in_ch + growth_rate, growth_rate),
        )
        self.encoder_2 = nn.Sequential(
            DenseDownsample(in_ch + 2 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 3 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 4 * growth_rate, growth_rate)
        )
        self.encoder_3 = nn.Sequential(
            DenseDownsample(in_ch + 5 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 6 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 7 * growth_rate, growth_rate)
        )
        self.encoder_4 = nn.Sequential(
            DenseDownsample(in_ch + 8 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 9 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 10 * growth_rate, growth_rate)
        )
        self.encoder_5 = nn.Sequential(
            DenseDownsample(in_ch + 11 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 12 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 13 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 14 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 15 * growth_rate, growth_rate)
        )
    def forward(self, x):
        out_encoder_1 = self.encoder_1(x)
        out_encoder_2 = self.encoder_2(out_encoder_1)
        out_encoder_3 = self.encoder_3(out_encoder_2)
        out_encoder_4 = self.encoder_4(out_encoder_3)
        out_encoder_5 = self.encoder_5(out_encoder_4)
        return [out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5]


class Decoder(nn.Module):
    def __init__(self, in_ch, growth_rate, upsample_chan, out_ch):
        super(Decoder, self).__init__()
        self.upconv_4 = UNetUpsample(in_ch + 16 * growth_rate, upsample_chan)
        self.decoder_conv_4 = nn.Sequential(
            SingleConv(in_ch + 11 * growth_rate + upsample_chan, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            SingleConv(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        )
        self.upconv_3 = UNetUpsample(256, upsample_chan)
        self.decoder_conv_3 = nn.Sequential(
            SingleConv(in_ch + 8 * growth_rate + upsample_chan, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            SingleConv(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        )
        self.upconv_2 = UNetUpsample(128, upsample_chan)
        self.decoder_conv_2 = nn.Sequential(
            SingleConv(in_ch + 5 * growth_rate + upsample_chan, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            SingleConv(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        )
        self.upconv_1 = UNetUpsample(64, upsample_chan)
        self.decoder_conv_1 = nn.Sequential(
            SingleConv(in_ch + 2 * growth_rate + upsample_chan, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            SingleConv(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        )
        self.final_conv = nn.Conv3d(32, out_ch, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=True)

    def forward(self, out_encoder):
        out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5 = out_encoder
        out_decoder_4 = self.decoder_conv_4(torch.cat((self.upconv_4(out_encoder_5), out_encoder_4), dim=1))
        out_decoder_3 = self.decoder_conv_3(torch.cat((self.upconv_3(out_decoder_4), out_encoder_3), dim=1))
        out_decoder_2 = self.decoder_conv_2(torch.cat((self.upconv_2(out_decoder_3), out_encoder_2), dim=1))
        out_decoder_1 = self.decoder_conv_1(torch.cat((self.upconv_1(out_decoder_2), out_encoder_1), dim=1))
        final_output = self.final_conv(out_decoder_1)
        return final_output


class HD_UNet(nn.Module):
    def __init__(self, in_ch, growth_rate, upsample_chan, out_ch):
        super(HD_UNet, self).__init__()
        self.encoder = Encoder(in_ch, growth_rate)
        self.decoder = Decoder(in_ch, growth_rate, upsample_chan, out_ch)
        self.initialize()

    @staticmethod
    def init_conv_IN(modules):
        for m in modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.InstanceNorm3d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def initialize(self):
        print('# random init decoder weight using nn.init.kaiming_uniform !')
        self.init_conv_IN(self.decoder.modules)
        print('# random init encoder weight using nn.init.kaiming_uniform !')
        self.init_conv_IN(self.encoder.modules)

    def forward(self, x):
        out_encoder = self.encoder(x)
        out_decoder = self.decoder(out_encoder)
        return out_decoder


class Model(nn.Module):
    def __init__(self, in_ch, growth_rate, upsample_chan, out_ch):
        super(Model, self).__init__()
        self.model = HD_UNet(in_ch, growth_rate, upsample_chan, out_ch)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.model(x))


# ------------------------------
#   Global Parameters
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

channel_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1] 
batch_size = 8  # 3D volumes are large, use small batch size
total_epochs = 700
check_epoch = 5
CHUNK_SIZE = 20  # Number of samples to load at once for training (adjust based on memory)
PATCH_SIZE = (32, 256, 256) # Depth, Height, Width
val_size = 0.2

clip_min = -300
clip_max = 800
prescription_dose = 26.00
target_dose = prescription_dose * 1.1

data_root = os.environ.get("WBI_DATA_ROOT", "./data")
train_ct_contour_dir = os.path.join(data_root, "Train", "CT_and_Contour")
train_dose_dir = os.path.join(data_root, "Train", "Dose")
save_file = './Model_Weight_HD_3D_UNet/best_model_weight_stage1.pth'
save_check_file = os.path.join(os.environ.get("WBI_OUTPUT_ROOT", "./outputs"), "epoch_image_3d_1stage")

# ------------------------------
#   Dataset (True 3D Patch-based)
# ------------------------------
class CustomDataset3D(Dataset):
    def __init__(self, ct_contour_paths, dose_paths, patch_size=(32, 256, 256), augment=False):
        self.ct_contour_paths = ct_contour_paths
        self.dose_paths = dose_paths
        self.patch_size = patch_size
        self.augment = augment
        
    def __len__(self):
        return len(self.ct_contour_paths)

    def __getitem__(self, idx):
        ct_contour_data = np.load(self.ct_contour_paths[idx], mmap_mode='r') # (H, W, Z, C)
        dose_data = np.load(self.dose_paths[idx], mmap_mode='r') # (H, W, Z)
        H, W, Z, C = ct_contour_data.shape
        D_p, H_p, W_p = self.patch_size

        # Patch extraction along Z (Depth)
        if self.augment and Z > D_p:
            # 1. CTV가 포함된 슬라이스 인덱스 찾기 (Channel 2: CTV)
            ctv_mask = ct_contour_data[:, :, :, 2]
            ctv_z_sum = np.sum(ctv_mask, axis=(0, 1))
            ctv_indices = np.where(ctv_z_sum > 0)[0]

            if len(ctv_indices) > 0:
                # 2. CTV 범위 내에서 무작위 중심 슬라이스 선택
                z_min, z_max = int(ctv_indices.min()), int(ctv_indices.max())
                center_z = random.randint(z_min, z_max)

                # 3. Jitter 추가 (패치 절반 크기 범위 내에서 무작위 이동)
                jitter = random.randint(-D_p // 4, D_p // 4)
                start_z = center_z - D_p // 2 + jitter

                # 4. 전체 볼륨 범위 내로 제한
                start_z = max(0, min(start_z, Z - D_p))
            else:
                # CTV가 없는 경우 (데이터 오류 등) 무작위 추출
                start_z = random.randint(0, Z - D_p)
        else:
            # 검증 단계: 전체 볼륨의 중앙 또는 패치 중심 추출
            start_z = max(0, (Z - D_p) // 2)

        # Extraction
        # Note: Input .npy is (H, W, Z, C)

        # We want (C, D, H, W)
        ct_patch = ct_contour_data[:, :, start_z:start_z+D_p, :].copy()
        dose_patch = dose_data[:, :, start_z:start_z+D_p].copy()
        
        # Preprocessing: Clip
        ct_patch = np.clip(ct_patch, clip_min, clip_max)
        
        # Transpose to (C, D, H, W)
        x = np.transpose(ct_patch, (3, 2, 0, 1)).astype(np.float32)
        # dose_patch is (H, W, D), we want (1, D, H, W)
        y = np.transpose(dose_patch, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        # Data Augmentation
        if self.augment:
            # 1. Rotation (-8, -5, -2, 0, 2, 5, 8 degrees)
            angle = random.choice([-8, -5, -2, 0, 2, 5, 8])
            if angle != 0:
                # Rotate CT and Masks separately to use different fill values
                # CT (channel 0) fill with clip_min (-300)
                ct_rot = TF.rotate(x[0:1], angle, fill=float(clip_min))
                # Masks (channels 1-8) fill with 0
                masks_rot = TF.rotate(x[1:], angle, fill=0.0)
                x = torch.cat([ct_rot, masks_rot], dim=0)
                
                # Dose (y) fill with 0
                y = TF.rotate(y, angle, fill=0.0)
            
            # 2. CT Intensity Augmentation (Channel 0)
            # Randomly shift and scale CT values
            shift = random.uniform(-20, 20)
            scale = random.uniform(0.95, 1.05)
            x[0] = (x[0] + shift) * scale
            x[0] = torch.clamp(x[0], clip_min, clip_max)
        
        # ROI masks for DVH loss (Created after augmentation)
        roi_masks = {
            "A_LAD": x[1:2].clone(),
            "CTV": x[2:3].clone(),
            "Contra_Breast": x[3:4].clone(),
            "External": x[4:5].clone(),
            "Heart": x[5:6].clone(),
            "Ipsi_Lung": x[6:7].clone(),
            "Contra_Lung": x[7:8].clone(),
            "RING": x[8:9].clone(),
        }
        
        return x, y, roi_masks

# ------------------------------
#   3D Loss Functions
# ------------------------------
def _safe_div(a, b, eps=1e-6):
    return a / (b + eps)

def soft_vx_percent(dose_gy, mask01, thr_gy, tau=0.25):
    p = torch.sigmoid((dose_gy - thr_gy) / tau)
    # Sum over spatial dims (D, H, W) -> dims (2, 3, 4)
    frac = _safe_div((p * mask01).sum(dim=(2,3,4)), mask01.sum(dim=(2,3,4)))
    return frac * 100.0

def mean_dose_gy(dose_gy, mask01):
    s = (dose_gy * mask01).sum(dim=(2,3,4))
    v = mask01.sum(dim=(2,3,4))
    return _safe_div(s, v)

def soft_dmax_gy(dose_gy, mask01, tau_max=0.5):
    B = dose_gy.shape[0]
    x = dose_gy.view(B, -1)
    m = mask01.view(B, -1)
    
    # Check if mask exists for each sample in batch
    mask_exists = (m.sum(dim=1) > 1e-5)
    
    # Use a large negative number that won't cause overflow during division
    neg_inf = -1e9
    x_masked = torch.where(m > 0.5, x, torch.tensor(neg_inf, device=x.device, dtype=x.dtype))
    
    # logsumexp calculation
    # To avoid -inf which leads to nan in subtraction (-inf - -inf = nan),
    # we ensure we don't have all-neg_inf rows before logsumexp or handle it after.
    val = tau_max * torch.logsumexp(x_masked / tau_max, dim=1)
    
    # For samples with no mask, return 0 instead of -inf to avoid nan in (pred_vmax - true_vmax)
    return torch.where(mask_exists, val, torch.zeros_like(val))

class DVHConstraintLoss3D(nn.Module):
    def __init__(self, rx_gy=26.0, target_factor=1.1, tau=0.25, tau_max=0.5, clinical=None, strict=None, weights=None, gt_offsets=None):
        super().__init__()
        self.rx_gy = float(rx_gy)
        self.target_factor = float(target_factor)
        self.tau = float(tau)
        self.tau_max = float(tau_max)
        
        self.clinical = clinical or {
            "CTV_V95_min_pct": 95.0, "CTV_Vmax_max_gy": 1.07 * self.rx_gy,
            "Heart_V7_max_pct": 3.0, "Heart_V1p5_max_pct": 30.0,
            "IpsiLung_V8_max_pct": 15.0, "ContraLung_Dmean_max_gy": 2.0,
            "ContraBreast_Dmean_max_gy": 3.0,
        }
        self.strict = strict or {
            "CTV_V95_min_pct": 97.0, "CTV_Vmax_max_gy": 1.05 * self.rx_gy,
            "Heart_V7_max_pct": 2.5, "Heart_V1p5_max_pct": 25.0,
            "IpsiLung_V8_max_pct": 13.0, "ContraLung_Dmean_max_gy": 1.8,
            "ContraBreast_Dmean_max_gy": 2.8,
        }
        self.weights = weights or {
            "CTV_V95": 20.0, "CTV_Vmax": 20.0, "Heart_V7": 2.0, "Heart_V1p5": 10.0,
            "IpsiLung_V8": 10.0, "ContraLung_Dmean": 5.5, "ContraBreast_Dmean": 5.5,
        }
        self.gt_offsets = gt_offsets or {
            "CTV_V95_pct": 1.0, "CTV_Vmax_gy": 0.0, "Heart_V7_pct": 1.5,
            "Heart_V1p5_pct": 8.0, "IpsiLung_V8_pct": 7.0, "ContraLung_Dmean_gy": 2.0,
            "ContraBreast_Dmean_gy": 2.0,
        }

    def forward(self, y_pred_norm, y_true_norm, roi_masks, meter=None):
        pred_gy = y_pred_norm * (self.rx_gy * self.target_factor)
        true_gy = y_true_norm * (self.rx_gy * self.target_factor)
        
        loss = torch.zeros([], device=pred_gy.device)
        
        # 1. CTV
        if "CTV" in roi_masks and roi_masks["CTV"].sum() > 0:
            m = roi_masks["CTV"]
            thr_v95 = 0.95 * self.rx_gy
            pred_v95 = soft_vx_percent(pred_gy, m, thr_v95, self.tau)
            true_v95 = soft_vx_percent(true_gy, m, thr_v95, self.tau)
            
            # Patch-wise GT가 임상 기준(95%)을 만족할 때만 strict 페널티(97%)를 고려함
            is_gt_satisfied = (true_v95 >= self.clinical["CTV_V95_min_pct"]).float()
            p_strict = F.relu(self.strict["CTV_V95_min_pct"] - pred_v95) * is_gt_satisfied
            p_gt = F.relu((true_v95 - self.gt_offsets["CTV_V95_pct"]) - pred_v95)
            
            final_p = torch.max(p_strict, p_gt)
            loss = loss + self.weights["CTV_V95"] * final_p.mean()
            if meter: meter.update("DVH_CTV_V95", final_p)

            pred_vmax = soft_dmax_gy(pred_gy, m, self.tau_max)
            true_vmax = soft_dmax_gy(true_gy, m, self.tau_max)
            # GT가 기준을 만족할 때만 strict 페널티 적용
            is_gt_satisfied_max = (true_vmax <= self.clinical["CTV_max_max_gy"]).float() if "CTV_max_max_gy" in self.clinical else (true_vmax <= self.clinical["CTV_Vmax_max_gy"]).float()
            p_strict_max = F.relu(pred_vmax - self.strict["CTV_Vmax_max_gy"]) * is_gt_satisfied_max
            p_gt_max = F.relu(pred_vmax - (true_vmax + self.gt_offsets["CTV_Vmax_gy"]))
            
            final_p_max = torch.max(p_strict_max, p_gt_max)
            loss = loss + self.weights["CTV_Vmax"] * final_p_max.mean()
            if meter: meter.update("DVH_CTV_Vmax", final_p_max)

        # 2. Heart (V7 and V1.5)
        if "Heart" in roi_masks and roi_masks["Heart"].sum() > 0:
            m = roi_masks["Heart"]
            # V7
            pred_v7 = soft_vx_percent(pred_gy, m, 7.0, self.tau)
            true_v7 = soft_vx_percent(true_gy, m, 7.0, self.tau)
            is_gt_satisfied_h7 = (true_v7 <= self.clinical["Heart_V7_max_pct"]).float()
            loss = loss + self.weights["Heart_V7"] * (F.relu(pred_v7 - (true_v7 + self.gt_offsets["Heart_V7_pct"])) + \
                   F.relu(pred_v7 - self.strict["Heart_V7_max_pct"]) * is_gt_satisfied_h7).mean()
            
            # V1.5 (Low dose suppression)
            pred_v1p5 = soft_vx_percent(pred_gy, m, 1.5, self.tau)
            true_v1p5 = soft_vx_percent(true_gy, m, 1.5, self.tau)
            is_gt_satisfied_h1p5 = (true_v1p5 <= self.clinical["Heart_V1p5_max_pct"]).float()
            loss = loss + self.weights["Heart_V1p5"] * (F.relu(pred_v1p5 - (true_v1p5 + self.gt_offsets["Heart_V1p5_pct"])) + \
                   F.relu(pred_v1p5 - self.strict["Heart_V1p5_max_pct"]) * is_gt_satisfied_h1p5).mean()

        # 3. Ipsi_Lung V8
        if "Ipsi_Lung" in roi_masks and roi_masks["Ipsi_Lung"].sum() > 0:
            m = roi_masks["Ipsi_Lung"]
            pred_v8 = soft_vx_percent(pred_gy, m, 8.0, self.tau)
            true_v8 = soft_vx_percent(true_gy, m, 8.0, self.tau)
            is_gt_satisfied_l8 = (true_v8 <= self.clinical["IpsiLung_V8_max_pct"]).float()
            loss = loss + self.weights["IpsiLung_V8"] * (F.relu(pred_v8 - (true_v8 + self.gt_offsets["IpsiLung_V8_pct"])) + \
                   F.relu(pred_v8 - self.strict["IpsiLung_V8_max_pct"]) * is_gt_satisfied_l8).mean()

        # 4. Contra_Lung Mean Dose
        if "Contra_Lung" in roi_masks and roi_masks["Contra_Lung"].sum() > 0:
            m = roi_masks["Contra_Lung"]
            pred_mean = mean_dose_gy(pred_gy, m)
            true_mean = mean_dose_gy(true_gy, m)
            is_gt_satisfied_cl = (true_mean <= self.clinical["ContraLung_Dmean_max_gy"]).float()
            loss = loss + self.weights["ContraLung_Dmean"] * (F.relu(pred_mean - (true_mean + self.gt_offsets["ContraLung_Dmean_gy"])) + \
                   F.relu(pred_mean - self.strict["ContraLung_Dmean_max_gy"]) * is_gt_satisfied_cl).mean()

        # 5. Contra_Breast Mean Dose
        if "Contra_Breast" in roi_masks and roi_masks["Contra_Breast"].sum() > 0:
            m = roi_masks["Contra_Breast"]
            pred_mean_cb = mean_dose_gy(pred_gy, m)
            true_mean_cb = mean_dose_gy(true_gy, m)
            is_gt_satisfied_cb = (true_mean_cb <= self.clinical["ContraBreast_Dmean_max_gy"]).float()
            loss = loss + self.weights["ContraBreast_Dmean"] * (F.relu(pred_mean_cb - (true_mean_cb + self.gt_offsets["ContraBreast_Dmean_gy"])) + \
                   F.relu(pred_mean_cb - self.strict["ContraBreast_Dmean_max_gy"]) * is_gt_satisfied_cb).mean()
            
        return loss

class SSIM3DLoss(nn.Module):
    def __init__(self, data_range=1.0):
        super().__init__()
        self.ssim = SSIM(data_range=data_range, size_average=True, channel=1, spatial_dims=2)

    def forward(self, x, y):
        # x, y: (B, 1, D, H, W)
        B, C, D, H, W = x.shape
        x_2d = x.transpose(1, 2).reshape(B * D, C, H, W)
        y_2d = y.transpose(1, 2).reshape(B * D, C, H, W)
        return 1 - self.ssim(x_2d, y_2d)

ssim_loss_fn = SSIM3DLoss(data_range=target_dose)
dvh_loss_fn = DVHConstraintLoss3D(rx_gy=prescription_dose)

def total_loss(y_true, y_pred, roi_masks, meter=None):
    y_pred_gy = y_pred * target_dose
    y_true_gy = y_true * target_dose

    alpha = 0.90
    l1_loss = F.l1_loss(y_pred_gy, y_true_gy)
    ssim_loss = ssim_loss_fn(y_pred_gy, y_true_gy)
    base_loss = (alpha * l1_loss) + ((1 - alpha) * ssim_loss)

    dvh_penalty = dvh_loss_fn(y_pred, y_true, roi_masks, meter=meter)

    total = (base_loss * 3.5) + (dvh_penalty * 0.0)
    if meter:
        meter.update("L1", l1_loss)
        meter.update("SSIM", ssim_loss)
        meter.update("DVH", dvh_penalty)
        meter.update("TotalLoss", total)
    return total

# ------------------------------
#   Visualization & Utils
# ------------------------------
class AvgMeter:
    def __init__(self):
        self.sum = defaultdict(float)
        self.cnt = defaultdict(int)
    def update(self, key, value):
        if hasattr(value, "detach"): value = value.detach().float().mean().item()
        self.sum[key] += float(value)
        self.cnt[key] += 1
    def avg(self, key):
        return self.sum[key] / self.cnt[key] if self.cnt[key] > 0 else 0.0

def calculate_dvh(doses, bins):
    if doses.size == 0: return None, None
    hist, bin_edges = np.histogram(doses, bins=bins)
    if hist.sum() == 0: return bin_edges[1:], np.zeros_like(bin_edges[1:])
    dvh = 100.0 * np.cumsum(hist[::-1])[::-1] / hist.sum()
    return bin_edges[1:], dvh

def infer_full_volume(model, ct_path, patch_size=(32, 256, 256)):
    model.eval()
    ct_contour_data = np.load(ct_path) # (H, W, Z, C)
    H, W, Z, C = ct_contour_data.shape
    D_p, H_p, W_p = patch_size
    
    # Preprocessing
    ct_processed = np.clip(ct_contour_data, clip_min, clip_max)
    
    full_pred = np.zeros((Z, H, W), dtype=np.float32)
    count_map = np.zeros((Z, H, W), dtype=np.float32)
    
    with torch.no_grad():
        # Sliding window along Z (Depth)
        for z in range(0, Z, D_p // 2): # Use overlap for smoother results
            start_z = z
            end_z = start_z + D_p
            if end_z > Z:
                end_z = Z
                start_z = max(0, end_z - D_p)
            
            patch = ct_processed[:, :, start_z:end_z, :]
            x = np.transpose(patch, (3, 2, 0, 1))[np.newaxis, ...].astype(np.float32)
            x = torch.from_numpy(x).to(device)
            
            y_pred = model(x).cpu().squeeze(0).squeeze(0).numpy() # (D_p, H, W)
            
            full_pred[start_z:end_z] += y_pred
            count_map[start_z:end_z] += 1
            
            if end_z == Z: break
            
    return full_pred / (count_map + 1e-8)

def visualize_prediction(y_true, y_pred, x_input, epoch, output_dir, case_name=""):
    # y_true, y_pred: (Z, H, W) numpy - Full Volume
    # x_input: (C, Z, H, W) numpy - Full Volume
    os.makedirs(output_dir, exist_ok=True)
    
    y_true_gy = y_true * target_dose
    y_pred_gy = y_pred * target_dose
    
    # Find the global max dose slice in ground truth
    max_idx = np.unravel_index(np.argmax(y_true_gy), y_true_gy.shape)
    center_d = max_idx[0]
    
    true_slice = y_true_gy[center_d]
    pred_slice = y_pred_gy[center_d]
    ct_slice = x_input[0, center_d]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # CT
    axes[0,0].imshow(ct_slice, cmap='gray'); axes[0,0].set_title(f"CT (Slice {center_d})")
    # True Dose
    im1 = axes[0,1].imshow(true_slice, cmap='jet', vmin=0, vmax=target_dose)
    plt.colorbar(im1, ax=axes[0,1]); axes[0,1].set_title("True Dose (Gy)")
    # Pred Dose
    im2 = axes[0,2].imshow(pred_slice, cmap='jet', vmin=0, vmax=target_dose)
    plt.colorbar(im2, ax=axes[0,2]); axes[0,2].set_title("Pred Dose (Gy)")
    
    # Diff
    diff = np.abs(true_slice - pred_slice)
    im3 = axes[1,0].imshow(diff, cmap='hot', vmin=0, vmax=5)
    plt.colorbar(im3, ax=axes[1,0]); axes[1,0].set_title("Diff (Gy)")
    
    # Full Volume DVH
    ax_dvh = axes[1,1]
    bins = np.linspace(0, target_dose, 100)
    # ROIs: 1:A_LAD, 2:CTV, 3:Contra_Breast, 5:Heart, 6:Ipsi_Lung, 7:Contra_Lung
    roi_info = {
        "CTV": {"idx": 2, "color": 'blue'},
        "Heart": {"idx": 5, "color": 'green'},
        "Ipsi_Lung": {"idx": 6, "color": 'cyan'},
        "A_LAD": {"idx": 1, "color": 'red'},
        "Contra_Breast": {"idx": 3, "color": 'magenta'},
        "Contra_Lung": {"idx": 7, "color": 'orange'},
    }
    
    for name, info in roi_info.items():
        mask = x_input[info["idx"]] > 0.5
        if mask.sum() > 0:
            b, t_dvh = calculate_dvh(y_true_gy[mask], bins)
            _, p_dvh = calculate_dvh(y_pred_gy[mask], bins)
            ax_dvh.plot(b, t_dvh, color=info["color"], linestyle='-', alpha=0.6, label=f"{name} True")
            ax_dvh.plot(b, p_dvh, color=info["color"], linestyle='--', linewidth=2, label=f"{name} Pred")

    # Add Clinical Constraints Markers
    # 1. CTV: V95% >= 95% (26Gy * 0.95 = 24.7Gy)
    v95_dose = prescription_dose * 0.95
    ax_dvh.plot(v95_dose, 95, 'ro', markersize=8, label="Constraint")
    ax_dvh.axvline(x=v95_dose, color='gray', linestyle=':', alpha=0.5)
    ax_dvh.axhline(y=95, color='gray', linestyle=':', alpha=0.5)
    
    # 2. Heart: V7Gy <= 3%, V1.5Gy <= 30%
    ax_dvh.plot(7.0, 3, 'rx', markersize=8)
    ax_dvh.plot(1.5, 30, 'rx', markersize=8)
    
    # 3. Ipsi_Lung: V8Gy <= 15%
    ax_dvh.plot(8.0, 15, 'r*', markersize=8)

    ax_dvh.set_title("Full Volume DVH & Clinical Constraints")
    ax_dvh.set_xlabel("Dose (Gy)"); ax_dvh.set_ylabel("Volume (%)")
    ax_dvh.set_xlim(0, target_dose); ax_dvh.set_ylim(0, 105)
    ax_dvh.legend(fontsize='small', loc='upper right', ncol=2)
    ax_dvh.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Hide the empty 6th subplot
    axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"epoch_{epoch}_{case_name}.png"))
    plt.close()

# ------------------------------
#   Main Training Loop
# ------------------------------
def main():
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    os.makedirs(save_check_file, exist_ok=True)
    
    in_channels = 9
    model = Model(in_ch=in_channels, growth_rate=16, upsample_chan=32, out_ch=1).to(device)

    # Load existing weights if available
    if os.path.exists(save_file):
        print(f"# Loading existing weights from {save_file}...")
        try:
            model.load_state_dict(torch.load(save_file, map_location=device))
            print("# Weights loaded successfully!")
        except Exception as e:
            print(f"# Error loading weights: {e}")
            print("# Starting from scratch.")
    else:
        print("# No existing weights found. Starting from scratch.")
    
    # Manual shape check
    print("\n# Verifying model input/output shapes...")
    test_input = torch.randn(1, in_channels, PATCH_SIZE[0], PATCH_SIZE[1], PATCH_SIZE[2]).to(device)
    with torch.no_grad():
        test_output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    print("# Model verification successful!\n")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    writer = SummaryWriter(log_dir=f"runs/HD_3D_UNet_stage1_{time.strftime('%Y%m%d_%H%M%S')}")

    train_files = sorted([os.path.join(train_ct_contour_dir, f) for f in os.listdir(train_ct_contour_dir) if f.endswith('.npy')])
    train_dose_files = sorted([os.path.join(train_dose_dir, f) for f in os.listdir(train_dose_dir) if f.endswith('.npy')])
    
    t_ct, v_ct, t_dose, v_dose = train_test_split(train_files, train_dose_files, test_size=val_size, random_state=42)
    
    # Select WL and WR cases for fixed visualization
    v_wl_idx = next((i for i, f in enumerate(v_ct) if 'WL' in os.path.basename(f)), None)
    v_wr_idx = next((i for i, f in enumerate(v_ct) if 'WR' in os.path.basename(f)), None)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, total_epochs + 1):
        model.train()
        train_meter = AvgMeter()
        
        # Shuffle files
        t_ct_s, t_dose_s = shuffle(t_ct, t_dose)
        
        # Chunked loading
        for i in range(0, len(t_ct_s), CHUNK_SIZE):
            chunk_ct = t_ct_s[i:i+CHUNK_SIZE]
            chunk_dose = t_dose_s[i:i+CHUNK_SIZE]
            train_ds = CustomDataset3D(chunk_ct, chunk_dose, patch_size=PATCH_SIZE, augment=True)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            
            for x, y, masks in tqdm(train_loader, desc=f"Epoch {epoch} Chunk {i//CHUNK_SIZE + 1}"):
                x, y = x.to(device), y.to(device)
                for k in masks: masks[k] = masks[k].to(device)
                
                optimizer.zero_grad()
                y_pred = model(x)
                loss = total_loss(y, y_pred, masks, meter=train_meter)
                loss.backward()
                optimizer.step()
            
            del train_ds, train_loader
            gc.collect()

        # Validation
        model.eval()
        val_meter = AvgMeter()
        val_ds = CustomDataset3D(v_ct, v_dose, patch_size=PATCH_SIZE, augment=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for x, y, masks in val_loader:
                x, y = x.to(device), y.to(device)
                for k in masks: masks[k] = masks[k].to(device)
                y_pred = model(x)
                total_loss(y, y_pred, masks, meter=val_meter)
        
        avg_val_loss = val_meter.avg("TotalLoss")
        print(f"Epoch {epoch} | Train Loss: {train_meter.avg('TotalLoss'):.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Update Scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        writer.add_scalar("Loss/Train", train_meter.avg("TotalLoss"), epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("LR", current_lr, epoch)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_file)
            print("Saved Best Model")
            
        if epoch % check_epoch == 0:
            # Full Volume Visualization for WL and WR
            for idx, label in [(v_wl_idx, "val_WL"), (v_wr_idx, "val_WR")]:
                if idx is not None:
                    ct_path = v_ct[idx]
                    dose_path = v_dose[idx]
                    
                    # Full Inference
                    pred_full = infer_full_volume(model, ct_path, PATCH_SIZE)
                    
                    # Load original full data for visualization
                    ct_full = np.load(ct_path) # (H, W, Z, C)
                    dose_full = np.load(dose_path) # (H, W, Z)
                    
                    # Transpose for visualization (C, Z, H, W)
                    ct_vis = np.transpose(ct_full, (3, 2, 0, 1))
                    dose_vis = np.transpose(dose_full, (2, 0, 1))
                    
                    visualize_prediction(dose_vis, pred_full, ct_vis, epoch, save_check_file, label)

if __name__ == "__main__":
    main()
