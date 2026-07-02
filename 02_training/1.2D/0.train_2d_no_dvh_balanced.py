# 2025.01.24 unet2D_simam_2D_no_DVH_balanced
# 2.5D 입력을 단일 2D 입력으로 변경하고, DVH loss 관련 기능을 제거한 실행 스크립트
# + WL/WR를 학습/검증에서 최대한 1:1로 맞추기 위한 balanced chunk / balanced batch sampler 추가
# + Test 데이터셋 완전 분리 및 Train 데이터를 8:2로 Train/Validation 분할

import os
import gc
import math
import time
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter

from scipy.ndimage import rotate
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from skimage.transform import resize
from tqdm import tqdm

from pytorch_msssim import SSIM

# Optional utilities (원본 기능 최대 유지)
try:
    from torchviz import make_dot
except Exception:
    make_dot = None

try:
    from torchsummary import summary
except Exception:
    summary = None

from utils.visualization import visualize_prediction, save_gif_from_slices


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================================================
# Global parameters
# =========================================================
channel_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1]  # CT + 8 ROI 채널
batch_size = 16
total_epochs = 1000
check_epoch = 10
CHUNK_SIZE = 20              # 한 번에 학습에 사용하는 환자 파일 수 (WL/WR 5:5를 위해 짝수 권장)
CACHE_SIZE = 100             # slice cache size
num_workers = 8
smooth = 1.0
img_rows = 256
img_cols = 256
clip_min = -300
clip_max = 800
start_index = 0
prescription_dose = prescribed_dose = 26.00
target_value = target_dose = prescribed_dose * 1.1

# 8:2 분할을 위해 0.2로 설정 (Train 폴더 데이터의 20%를 Validation으로 사용)
val_size = 0.2               
patience = 50 # early stopping을 위한 patience

# 2D 설정: 중심 슬라이스만 사용
slice_window = 0

# WL/WR 균형 설정
BALANCE_TRAIN_BATCHES = True
BALANCE_VAL_TOTAL = True
OVERSAMPLE_MINORITY_SIDE = True

train_ct_contour_dir = '../final_dataset/Train/CT_and_Contour'
train_dose_dir = '../final_dataset/Train/Dose'
save_file = './Model_Weight_2D_SimAM/best_model_weight.pth'
save_check_file = 'epoch_image'
grad_cam_name = 'gradcam'


# =========================================================
# Utility meters / figures
# =========================================================
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


class SideCounter:
    def __init__(self):
        self.counts = defaultdict(int)

    def update(self, side: str, n: int):
        self.counts[side] += int(n)

    def get(self, side: str):
        return int(self.counts.get(side, 0))

    def ratio_text(self):
        wl = self.get("WL")
        wr = self.get("WR")
        total = wl + wr
        if total == 0:
            return "WL=0, WR=0"
        return f"WL={wl} ({wl/total:.1%}), WR={wr} ({wr/total:.1%})"


def make_qualitative_figure(ct2d, roi_dict_2d, true2d_gy, pred2d_gy, vmax):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    ax0, ax1, ax2, ax3 = axes

    ax0.imshow(ct2d, cmap="gray")
    ax0.set_title("Input CT")
    ax0.axis("off")

    ax1.imshow(ct2d, cmap="gray")
    for _, m in roi_dict_2d.items():
        if np.asarray(m).sum() > 0:
            ax1.contour(m, levels=[0.5], linewidths=1)
    ax1.set_title("ROI masks (contours)")
    ax1.axis("off")

    im2 = ax2.imshow(true2d_gy, vmin=0, vmax=vmax)
    ax2.set_title("True dose (Gy)")
    ax2.axis("off")
    fig.colorbar(im2, ax=ax2, fraction=0.046)

    im3 = ax3.imshow(pred2d_gy, vmin=0, vmax=vmax)
    ax3.set_title("Pred dose (Gy)")
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


# =========================================================
# Side / pairing helpers
# =========================================================
def get_side_from_path(path: str) -> str:
    name = os.path.basename(path)
    if 'WL' in name:
        return 'WL'
    if 'WR' in name:
        return 'WR'
    raise ValueError(f"Could not infer WL/WR from filename: {name}")


def make_pairs(ct_paths: List[str], dose_paths: List[str]) -> List[Tuple[str, str]]:
    assert len(ct_paths) == len(dose_paths), "CT/Dose pair count mismatch"
    pairs = []
    for ct_p, dose_p in zip(ct_paths, dose_paths):
        ct_side = get_side_from_path(ct_p)
        dose_side = get_side_from_path(dose_p)
        if ct_side != dose_side:
            raise ValueError(
                f"CT/Dose side mismatch: {os.path.basename(ct_p)} vs {os.path.basename(dose_p)}"
            )
        pairs.append((ct_p, dose_p))
    return pairs


def split_pairs_by_side(pairs: List[Tuple[str, str]]):
    wl_pairs = []
    wr_pairs = []
    for ct_p, dose_p in pairs:
        side = get_side_from_path(ct_p)
        if side == 'WL':
            wl_pairs.append((ct_p, dose_p))
        else:
            wr_pairs.append((ct_p, dose_p))
    return wl_pairs, wr_pairs


def unpack_pairs(pairs: List[Tuple[str, str]]):
    ct_paths = [p[0] for p in pairs]
    dose_paths = [p[1] for p in pairs]
    return ct_paths, dose_paths


def repeat_or_trim_to_length(items: List[Tuple[str, str]], target_len: int, rng: random.Random):
    if len(items) == 0:
        return []
    if len(items) == target_len:
        out = list(items)
        rng.shuffle(out)
        return out
    if len(items) > target_len:
        out = list(items)
        rng.shuffle(out)
        return out[:target_len]

    out = list(items)
    while len(out) < target_len:
        extra = list(items)
        rng.shuffle(extra)
        need = target_len - len(out)
        out.extend(extra[:need])
    rng.shuffle(out)
    return out


def build_balanced_epoch_chunks(
    train_pairs: List[Tuple[str, str]],
    chunk_size: int,
    epoch: int,
    oversample_minority: bool = True,
):
    if chunk_size % 2 != 0:
        raise ValueError(f"CHUNK_SIZE must be even for 1:1 WL/WR chunks. Got {chunk_size}.")

    wl_pairs, wr_pairs = split_pairs_by_side(train_pairs)
    if len(wl_pairs) == 0 or len(wr_pairs) == 0:
        raise ValueError("Both WL and WR training files are required for balanced chunking.")

    rng = random.Random(1000 + epoch)
    half = chunk_size // 2

    if oversample_minority:
        target_side_len = max(len(wl_pairs), len(wr_pairs))
        wl_epoch_pairs = repeat_or_trim_to_length(wl_pairs, target_side_len, rng)
        wr_epoch_pairs = repeat_or_trim_to_length(wr_pairs, target_side_len, rng)
    else:
        target_side_len = min(len(wl_pairs), len(wr_pairs))
        wl_epoch_pairs = repeat_or_trim_to_length(wl_pairs, target_side_len, rng)
        wr_epoch_pairs = repeat_or_trim_to_length(wr_pairs, target_side_len, rng)

    num_chunks = math.ceil(target_side_len / half)

    def take_chunk(side_pairs: List[Tuple[str, str]], chunk_idx: int):
        start = chunk_idx * half
        end = min(start + half, len(side_pairs))
        chunk = list(side_pairs[start:end])
        if len(chunk) < half:
            refill_rng = random.Random(9000 + epoch + chunk_idx)
            source = list(side_pairs)
            refill_rng.shuffle(source)
            need = half - len(chunk)
            chunk.extend(source[:need])
        return chunk

    chunks = []
    for chunk_idx in range(num_chunks):
        wl_chunk = take_chunk(wl_epoch_pairs, chunk_idx)
        wr_chunk = take_chunk(wr_epoch_pairs, chunk_idx)
        chunk_pairs = wl_chunk + wr_chunk
        rng.shuffle(chunk_pairs)
        chunks.append(chunk_pairs)

    return chunks, wl_pairs, wr_pairs


# =========================================================
# Data augmentation
# =========================================================
def get_augment_params() -> Tuple[bool, int]:
    do_flip = random.random() < 0.5
    if random.random() < 0.5:
        angle = random.choice((-10, -8, -5, -2, 2, 5, 8, 10))
    else:
        angle = 0
    return do_flip, angle


def apply_transforms(image, do_flip, angle):
    if do_flip:
        image = np.flip(image, axis=1).copy()

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


# =========================================================
# Loss / metrics
# =========================================================
def dice_coef(y_true, y_pred):
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2.0 * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


class SSIM2DLoss(nn.Module):
    def __init__(self, data_range=1.0, channel=1):
        super().__init__()
        self.ssim_loss = SSIM(
            data_range=data_range,
            size_average=True,
            channel=channel,
            spatial_dims=2,
        )

    def forward(self, x, y):
        return 1.0 - self.ssim_loss(x, y)


ssim_loss_fn = SSIM2DLoss(data_range=prescription_dose * 1.1, channel=1)


def total_loss(y_true, y_pred, meter: Optional[AvgMeter] = None):
    y_pred_gy = y_pred * target_dose
    y_true_gy = y_true * target_dose

    alpha = 0.85
    l1_loss = F.l1_loss(y_pred_gy, y_true_gy)
    ssim_loss = ssim_loss_fn(y_pred_gy, y_true_gy)
    base_loss = (alpha * l1_loss) + ((1.0 - alpha) * ssim_loss)

    scaled_total = base_loss * 3.5

    if meter is not None:
        meter.update("L1_Gy", l1_loss)
        meter.update("SSIM", ssim_loss)
        meter.update("BaseLoss", base_loss)
        meter.update("TotalLoss", scaled_total)

    return scaled_total


# =========================================================
# Model
# =========================================================
class SimAM(nn.Module):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        mean = torch.mean(x, dim=[2, 3], keepdim=True)
        var = torch.mean((x - mean) ** 2, dim=[2, 3], keepdim=True)
        norm_x = (x - mean) / torch.sqrt(var + self.epsilon)
        return x * torch.sigmoid(norm_x)


def conv_block(in_feat, out_feat):
    return nn.Sequential(
        nn.Conv2d(in_feat, out_feat, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_feat),
        SimAM(),
    )


def up_block(in_feat, out_feat):
    return nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)


class UNet2D_SimAM(nn.Module):
    def __init__(self, in_channels=9, out_channels=1):
        super().__init__()
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


# =========================================================
# Dataset (true 2D single-slice input)
# =========================================================
class CustomDataset(Dataset):
    def __init__(
        self,
        ct_contour_paths,
        dose_paths,
        channel_weights,
        max_cache_size=CACHE_SIZE,
        augment=False,
    ):
        self.ct_contour_paths = ct_contour_paths
        self.dose_paths = dose_paths
        self.channel_weights = channel_weights
        self.augment = augment
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.file_cache = {}
        self.slice_cache = {}
        self.max_cache_size = max_cache_size
        self.cache_keys = []

        self.indices = []
        self.index_sides = []
        self.file_sides = []
        self.file_slice_counts = []

        for file_idx, ct_path in enumerate(self.ct_contour_paths):
            try:
                side = get_side_from_path(ct_path)
                with open(ct_path, 'rb') as f:
                    version = np.lib.format.read_magic(f)
                    shape, _, _ = np.lib.format._read_array_header(f, version)
                    z_dim = shape[2]

                self.file_sides.append(side)
                self.file_slice_counts.append(int(z_dim))
                for z in range(z_dim):
                    self.indices.append((file_idx, z))
                    self.index_sides.append(side)
            except Exception as e:
                print(f"Warning: Could not read header from {ct_path}. Skipping file. Error: {e}")

        self.side_to_indices = {
            "WL": [i for i, s in enumerate(self.index_sides) if s == "WL"],
            "WR": [i for i, s in enumerate(self.index_sides) if s == "WR"],
        }

    def __len__(self):
        return len(self.indices)

    def side_counts(self):
        return {side: len(idxs) for side, idxs in self.side_to_indices.items()}

    def file_side_counts(self):
        return {
            "WL": sum(1 for s in self.file_sides if s == "WL"),
            "WR": sum(1 for s in self.file_sides if s == "WR"),
        }

    def __getitem__(self, idx):
        file_idx, z = self.indices[idx]
        cache_key = (file_idx, z)
        use_cache = not self.augment

        if use_cache and cache_key in self.slice_cache:
            return self.slice_cache[cache_key]

        if file_idx not in self.file_cache:
            ct_contour_data = np.load(self.ct_contour_paths[file_idx], mmap_mode='r')
            dose_data = np.load(self.dose_paths[file_idx], mmap_mode='r')
            self.file_cache[file_idx] = (ct_contour_data, dose_data)
        else:
            ct_contour_data, dose_data = self.file_cache[file_idx]

        image = ct_contour_data[:, :, z, :].copy()   # (H, W, C)
        dose = dose_data[:, :, z].copy()             # (H, W)
        side = self.file_sides[file_idx]

        do_flip = False
        angle = 0
        if self.augment:
            do_flip, angle = get_augment_params()
            image = apply_transforms(image, do_flip, angle)
            dose = apply_transforms(dose, do_flip, angle)
            image = _apply_ct_intensity_aug_2d(image)

        image = np.clip(image, self.clip_min, self.clip_max)
        for c_idx in range(image.shape[-1]):
            image[:, :, c_idx] *= self.channel_weights[c_idx]

        x = torch.from_numpy(np.transpose(image, (2, 0, 1)).astype(np.float32))
        y = torch.from_numpy(dose.astype(np.float32)[None, ...])

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

        result = (x, y, roi_masks, side)

        if use_cache:
            if len(self.slice_cache) >= self.max_cache_size:
                oldest_key = self.cache_keys.pop(0)
                if oldest_key in self.slice_cache:
                    del self.slice_cache[oldest_key]
            self.slice_cache[cache_key] = result
            self.cache_keys.append(cache_key)

        return result


# =========================================================
# Balanced side sampler / loader
# =========================================================
class BalancedSideBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset: CustomDataset, batch_size: int, seed: int = 42, drop_last: bool = False):
        if batch_size % 2 != 0:
            raise ValueError(f"batch_size must be even for WL/WR balanced batches. Got {batch_size}.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.half_batch = batch_size // 2
        self.seed = seed
        self.drop_last = drop_last

        self.wl_indices = list(dataset.side_to_indices.get("WL", []))
        self.wr_indices = list(dataset.side_to_indices.get("WR", []))

        if len(self.wl_indices) == 0 or len(self.wr_indices) == 0:
            raise ValueError("BalancedSideBatchSampler requires both WL and WR samples.")

        self.num_batches = max(
            math.ceil(len(self.wl_indices) / self.half_batch),
            math.ceil(len(self.wr_indices) / self.half_batch),
        )
        self.num_samples = self.num_batches * self.batch_size

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed)
        wl_pool = list(self.wl_indices)
        wr_pool = list(self.wr_indices)
        rng.shuffle(wl_pool)
        rng.shuffle(wr_pool)

        wl_ptr = 0
        wr_ptr = 0

        def take(pool, base_indices, ptr, n):
            out = []
            while len(out) < n:
                remain = len(pool) - ptr
                need = n - len(out)
                take_n = min(remain, need)
                if take_n > 0:
                    out.extend(pool[ptr:ptr + take_n])
                    ptr += take_n
                if len(out) < n:
                    pool = list(base_indices)
                    rng.shuffle(pool)
                    ptr = 0
            return out, pool, ptr

        for _ in range(self.num_batches):
            batch_wl, wl_pool, wl_ptr = take(wl_pool, self.wl_indices, wl_ptr, self.half_batch)
            batch_wr, wr_pool, wr_ptr = take(wr_pool, self.wr_indices, wr_ptr, self.half_batch)
            batch = batch_wl + batch_wr
            rng.shuffle(batch)
            yield batch


def make_loader(dataset, batch_size, is_train, epoch_seed, balance_sides=False):
    common_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if balance_sides and len(dataset.side_to_indices.get("WL", [])) > 0 and len(dataset.side_to_indices.get("WR", [])) > 0:
        batch_sampler = BalancedSideBatchSampler(dataset, batch_size=batch_size, seed=epoch_seed)
        loader = DataLoader(dataset, batch_sampler=batch_sampler, **common_kwargs)
        return loader, True

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        **common_kwargs,
    )
    return loader, False


# =========================================================
# Grad-CAM (원본 기능 유지용)
# =========================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.forward_handle = None
        self.backward_handle = None

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.forward_handle = module.register_forward_hook(self.save_activation)
                self.backward_handle = module.register_full_backward_hook(self.save_gradient)
                break

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor):
        self.model.eval()
        output = self.model(input_tensor)
        loss = output[:, 0].sum()
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        alpha = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (alpha * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cam / (np.max(cam) + 1e-8)
        cam = resize(cam, (img_rows, img_cols))
        return cam

    def release(self):
        if self.forward_handle:
            self.forward_handle.remove()
        if self.backward_handle:
            self.backward_handle.remove()


# =========================================================
# Visualization case preparation
# =========================================================
def prepare_single_test_case(ct_contour_path, dose_path, channel_weights):
    print(f"Preparing single visualization case from Validation set: {os.path.basename(ct_contour_path)}")

    ct_contour_data = np.load(ct_contour_path)  # (H, W, Z, 9)
    dose_data = np.load(dose_path)              # (H, W, Z)

    ct_contour_data = np.clip(ct_contour_data, clip_min, clip_max)

    inputs_list = []
    labels_list = []

    _, _, z_dim, c_dim = ct_contour_data.shape
    assert c_dim == 9, f"Expected 9 channels, got {c_dim}"

    for z in range(z_dim):
        slice_data = ct_contour_data[:, :, z, :].copy()
        for c_idx in range(c_dim):
            slice_data[:, :, c_idx] *= channel_weights[c_idx]

        x = np.transpose(slice_data, (2, 0, 1)).astype(np.float32)
        y = dose_data[:, :, z].astype(np.float32)[None, ...]
        inputs_list.append(x)
        labels_list.append(y)

    return np.array(inputs_list), np.array(labels_list)


# =========================================================
# Train / validation
# =========================================================
def train_model(
    model,
    optimizer,
    train_ct_contour_paths,
    train_dose_paths,
    total_epochs,
    batch_size,
    save_file,
    channel_weights,
    min_delta=0.001,
    patience=patience,
    writer=None,
    cache_size=CACHE_SIZE,
):
    # 레이블을 추출하여 층화추출(Stratified split)에 사용
    labels = ['WR' if 'WR' in os.path.basename(f) else 'WL' for f in train_ct_contour_paths]
    
    # Train 폴더 데이터를 지정된 비율(val_size=0.2)로 8:2 분할
    train_ct_paths, val_ct_paths, train_dose_paths, val_dose_paths = train_test_split(
        train_ct_contour_paths,
        train_dose_paths,
        test_size=val_size,
        random_state=42,
        stratify=labels,
    )

    train_pairs = make_pairs(train_ct_paths, train_dose_paths)
    val_pairs = make_pairs(val_ct_paths, val_dose_paths)
    train_pairs_wl, train_pairs_wr = split_pairs_by_side(train_pairs)
    val_pairs_wl, val_pairs_wr = split_pairs_by_side(val_pairs)

    val_ct_paths_wl, val_dose_paths_wl = unpack_pairs(val_pairs_wl)
    val_ct_paths_wr, val_dose_paths_wr = unpack_pairs(val_pairs_wr)

    print("\n--- Stratified Data Split ---")
    print(f"Total Training set: {len(train_ct_paths)} files (will be chunked per epoch)")
    print(f"  - Train WL files: {len(train_pairs_wl)}")
    print(f"  - Train WR files: {len(train_pairs_wr)}")
    print(f"Validation set (Total): {len(val_ct_paths)} files")
    print(f"  - Val WL files: {len(val_ct_paths_wl)}")
    print(f"  - Val WR files: {len(val_ct_paths_wr)}")
    print("-----------------------------\n")

    val_dataset = CustomDataset(val_ct_paths, val_dose_paths, channel_weights, max_cache_size=cache_size, augment=False)
    val_dataset_wl = CustomDataset(val_ct_paths_wl, val_dose_paths_wl, channel_weights, max_cache_size=cache_size, augment=False)
    val_dataset_wr = CustomDataset(val_ct_paths_wr, val_dose_paths_wr, channel_weights, max_cache_size=cache_size, augment=False)

    print("Validation slice composition:")
    print(f"  - Total slices: {val_dataset.side_counts()}")
    print(f"  - WL-only slices: {val_dataset_wl.side_counts()}")
    print(f"  - WR-only slices: {val_dataset_wr.side_counts()}")

    val_loader, val_total_balanced = make_loader(
        val_dataset,
        batch_size=batch_size,
        is_train=False,
        epoch_seed=4242,
        balance_sides=BALANCE_VAL_TOTAL,
    )
    val_loader_wl = DataLoader(
        val_dataset_wl,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    ) if len(val_dataset_wl) > 0 else None
    val_loader_wr = DataLoader(
        val_dataset_wr,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    ) if len(val_dataset_wr) > 0 else None

    print(f"Validation total loader balanced: {val_total_balanced}")

    # 시각화 데이터(Validation Set에서 추출)
    vis_cases = {}
    if val_ct_paths:
        wl_idx = next((i for i, p in enumerate(val_ct_paths) if 'WL' in os.path.basename(p)), -1)
        if wl_idx != -1:
            print("\n--- Preparing WL visualization case from Validation set ---")
            vis_cases['WL'] = {'data': prepare_single_test_case(val_ct_paths[wl_idx], val_dose_paths[wl_idx], channel_weights)}

        wr_idx = next((i for i, p in enumerate(val_ct_paths) if 'WR' in os.path.basename(p)), -1)
        if wr_idx != -1:
            print("\n--- Preparing WR visualization case from Validation set ---")
            vis_cases['WR'] = {'data': prepare_single_test_case(val_ct_paths[wr_idx], val_dose_paths[wr_idx], channel_weights)}

    if not vis_cases:
        print("Warning: No suitable WL/WR validation files found for visualization.")

    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop_triggered = False

    def evaluate_loader(loader):
        if loader is None:
            return 0.0, 0.0, 0.0, 0.0, {"WL": 0, "WR": 0}

        meter = AvgMeter()
        side_counter = SideCounter()
        model.eval()
        with torch.no_grad():
            for x_val, y_val, _, side_batch in loader:
                x_val = x_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True)
                y_pred_val = model(x_val)
                total_loss(y_val, y_pred_val, meter=meter)
                meter.update("Dice", dice_coef(y_val.detach().cpu(), y_pred_val.detach().cpu()))
                for side in side_batch:
                    side_counter.update(str(side), 1)
        return (
            meter.avg('TotalLoss'),
            meter.avg('Dice'),
            meter.avg('L1_Gy'),
            meter.avg('SSIM'),
            dict(side_counter.counts),
        )

    for epoch in tqdm(range(1, total_epochs + 1)):
        model.train()
        train_meter = AvgMeter()
        train_side_counter = SideCounter()

        epoch_chunks, train_wl_full, train_wr_full = build_balanced_epoch_chunks(
            train_pairs=train_pairs,
            chunk_size=CHUNK_SIZE,
            epoch=epoch,
            oversample_minority=OVERSAMPLE_MINORITY_SIDE,
        )

        print(
            f"\nEpoch {epoch}: training with {len(epoch_chunks)} balanced chunks "
            f"(base train files WL={len(train_wl_full)}, WR={len(train_wr_full)}, chunk_size={CHUNK_SIZE})"
        )

        for i, chunk_pairs in enumerate(epoch_chunks):
            ct_chunk, dose_chunk = unpack_pairs(chunk_pairs)
            train_dataset = CustomDataset(ct_chunk, dose_chunk, channel_weights, max_cache_size=cache_size, augment=True)
            train_loader, train_balanced = make_loader(
                train_dataset,
                batch_size=batch_size,
                is_train=True,
                epoch_seed=epoch * 1000 + i,
                balance_sides=BALANCE_TRAIN_BATCHES,
            )

            chunk_file_counts = train_dataset.file_side_counts()
            chunk_slice_counts = train_dataset.side_counts()
            print(
                f"  Chunk {i + 1}/{len(epoch_chunks)} | files WL={chunk_file_counts['WL']} WR={chunk_file_counts['WR']} "
                f"| slices WL={chunk_slice_counts['WL']} WR={chunk_slice_counts['WR']} "
                f"| balanced_batch_sampler={train_balanced}"
            )

            for x, y, _, side_batch in tqdm(train_loader, desc=f"Epoch {epoch}, Chunk {i + 1}/{len(epoch_chunks)}", leave=False):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                y_pred = model(x)
                loss = total_loss(y, y_pred, meter=train_meter)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_meter.update("Dice", dice_coef(y.detach().cpu(), y_pred.detach().cpu()))
                for side in side_batch:
                    train_side_counter.update(str(side), 1)

            del train_dataset, train_loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        val_loss_avg, val_dice_avg, val_l1_avg, val_ssim_avg, val_total_side_counts = evaluate_loader(val_loader)
        val_loss_wl, val_dice_wl, _, _, val_wl_side_counts = evaluate_loader(val_loader_wl)
        val_loss_wr, val_dice_wr, _, _, val_wr_side_counts = evaluate_loader(val_loader_wr)

        if writer is not None:
            writer.add_scalar('Loss/Train', train_meter.avg('TotalLoss'), epoch)
            writer.add_scalar('Loss/Train_L1_Gy', train_meter.avg('L1_Gy'), epoch)
            writer.add_scalar('Loss/Train_SSIM', train_meter.avg('SSIM'), epoch)
            writer.add_scalar('Dice/Train', train_meter.avg('Dice'), epoch)
            writer.add_scalar('Count/Train_WL_Slices', train_side_counter.get('WL'), epoch)
            writer.add_scalar('Count/Train_WR_Slices', train_side_counter.get('WR'), epoch)

            writer.add_scalar('Loss/Val_Total', val_loss_avg, epoch)
            writer.add_scalar('Loss/Val_L1_Gy', val_l1_avg, epoch)
            writer.add_scalar('Loss/Val_SSIM', val_ssim_avg, epoch)
            writer.add_scalar('Dice/Val_Total', val_dice_avg, epoch)
            writer.add_scalar('Count/ValTotal_WL_Slices', int(val_total_side_counts.get('WL', 0)), epoch)
            writer.add_scalar('Count/ValTotal_WR_Slices', int(val_total_side_counts.get('WR', 0)), epoch)

            writer.add_scalar('Loss/Val_WL', val_loss_wl, epoch)
            writer.add_scalar('Dice/Val_WL', val_dice_wl, epoch)
            writer.add_scalar('Count/ValWL_Slices', int(val_wl_side_counts.get('WL', 0)), epoch)

            writer.add_scalar('Loss/Val_WR', val_loss_wr, epoch)
            writer.add_scalar('Dice/Val_WR', val_dice_wr, epoch)
            writer.add_scalar('Count/ValWR_Slices', int(val_wr_side_counts.get('WR', 0)), epoch)

        print(
            f"\nEpoch [{epoch}/{total_epochs}] - "
            f"Train Loss: {train_meter.avg('TotalLoss'):.4f}, "
            f"Val Loss: {val_loss_avg:.4f}, "
            f"Train Dice: {train_meter.avg('Dice'):.4f}, "
            f"Val Dice: {val_dice_avg:.4f} "
            f"| Val(WL): {val_loss_wl:.4f}, Val(WR): {val_loss_wr:.4f}"
        )
        print(f"  Train sampled slices: {train_side_counter.ratio_text()}")
        print(
            f"  Val sampled slices: WL={int(val_total_side_counts.get('WL', 0))}, "
            f"WR={int(val_total_side_counts.get('WR', 0))}"
        )

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
            if vis_cases:
                for case_type, case_info in vis_cases.items():
                    print(f"--- Generating visualization for {case_type} case ---")
                    vis_inputs_np, vis_labels_np = case_info['data']
                    model.eval()
                    vis_inputs_tensor = torch.from_numpy(vis_inputs_np)
                    pred_chunks = []
                    with torch.no_grad():
                        for i in range(0, len(vis_inputs_tensor), batch_size):
                            bx = vis_inputs_tensor[i:i + batch_size].to(device, non_blocking=True)
                            pred = model(bx)
                            pred_chunks.append(pred.detach().cpu().numpy())

                    pred_vol = np.concatenate(pred_chunks, axis=0).squeeze(1)
                    true_vol = vis_labels_np.squeeze(1)
                    all_trues_gy = true_vol * target_dose
                    all_preds_gy = pred_vol * target_dose

                    if writer is not None:
                        z_valid = all_trues_gy.shape[0]
                        mid = z_valid // 2
                        t = all_trues_gy[mid]
                        p = all_preds_gy[mid]
                        ct2d = vis_inputs_np[mid, 0]
                        roi_dict_2d = {
                            "CTV": (vis_inputs_np[mid, 2] > 0.5).astype(np.uint8),
                            "Heart": (vis_inputs_np[mid, 5] > 0.5).astype(np.uint8),
                            "Ipsi_Lung": (vis_inputs_np[mid, 6] > 0.5).astype(np.uint8),
                            "Contra_Lung": (vis_inputs_np[mid, 7] > 0.5).astype(np.uint8),
                        }

                        fig_qual = make_qualitative_figure(ct2d, roi_dict_2d, t, p, vmax=target_dose)
                        writer.add_figure(f"Vis/{case_type}_qualitative", fig_qual, global_step=epoch)
                        fig_qual.savefig(os.path.join(save_check_file, f"epoch_{epoch}_{case_type}_qualitative.png"))
                        plt.close(fig_qual)

                        fig_diff = make_diff_figure(t, p)
                        writer.add_figure(f"Vis/{case_type}_diff", fig_diff, global_step=epoch)
                        fig_diff.savefig(os.path.join(save_check_file, f"epoch_{epoch}_{case_type}_diff.png"))
                        plt.close(fig_diff)

                        gif_output_path = os.path.join(save_check_file, f"epoch_{epoch}_{case_type}_dose_comparison.gif")
                        save_gif_from_slices(
                            arrays=[all_trues_gy, all_preds_gy],
                            titles=["Ground Truth", "Prediction"],
                            out_path=gif_output_path,
                            vmin=0,
                            vmax=target_dose,
                            duration=100,
                        )

                        visualize_prediction(
                            y_true=true_vol,
                            y_pred=pred_vol,
                            epoch=epoch,
                            output_dir=save_check_file,
                            start_index=start_index,
                            x_input=vis_inputs_np,
                        )

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                model.train()

        if early_stop_triggered:
            print("Exiting training loop due to early stopping.")
            break

    print("Training process finished.")


# =========================================================
# Main
# =========================================================
def main():
    if CHUNK_SIZE % 2 != 0:
        raise ValueError(f"CHUNK_SIZE must be even for WL/WR 1:1 chunking. Current value: {CHUNK_SIZE}")
    if batch_size % 2 != 0:
        raise ValueError(f"batch_size must be even for WL/WR 1:1 balanced batches. Current value: {batch_size}")

    save_dir = os.path.dirname(save_file)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_check_file, exist_ok=True)
    os.makedirs(grad_cam_name, exist_ok=True)

    in_channels = len(channel_weights)  # true 2D: 단일 slice 9채널
    model = UNet2D_SimAM(in_channels=in_channels, out_channels=1).to(device)

    x = torch.randn(1, in_channels, img_rows, img_cols).to(device)
    y = model(x)

    if summary is not None:
        try:
            summary(model, (in_channels, img_rows, img_cols))
        except Exception as e:
            print(f"[Warning] torchsummary failed: {e}")

    if make_dot is not None:
        try:
            graph = make_dot(y, params=dict(model.named_parameters()))
            graph.render("Simam_model_2D_no_DVH_balanced", format='png')
        except Exception as e:
            print(f"[Warning] torchviz graph render failed: {e}")

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'bn' in name or 'norm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.Adam(
        [
            {'params': decay_params, 'weight_decay': 1e-6},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ],
        lr=1e-5,
    )

    latest_epoch = -1
    latest_model_path = None
    if os.path.isdir(save_dir) and os.path.exists(save_file):
        print(f"Loading from default path: {save_file}")
        model.load_state_dict(torch.load(save_file, map_location=device))
    else:
        print("Creating a new model...")

    # Train 데이터만 로딩
    train_files = sorted([
        os.path.join(train_ct_contour_dir, f)
        for f in os.listdir(train_ct_contour_dir)
        if f.endswith('.npy')
    ])
    train_dose_files = sorted([
        os.path.join(train_dose_dir, f)
        for f in os.listdir(train_dose_dir)
        if f.endswith('.npy')
    ])

    assert len(train_files) == len(train_dose_files), "Train CT/Dose file count mismatch"

    train_pair_preview = make_pairs(train_files, train_dose_files)
    full_train_wl, full_train_wr = split_pairs_by_side(train_pair_preview)
    print(f"Full train file composition before split: WL={len(full_train_wl)}, WR={len(full_train_wr)}")

    log_dir = os.path.join("runs", f"SimAM_2D_noDVH_balanced_{time.strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_dir=log_dir)
    print("TensorBoard logdir:", log_dir)

    # test 관련 파라미터 제외하고 호출
    train_model(
        model=model,
        optimizer=optimizer,
        train_ct_contour_paths=train_files,
        train_dose_paths=train_dose_files,
        total_epochs=total_epochs,
        batch_size=batch_size,
        save_file=save_file,
        channel_weights=channel_weights,
        writer=writer,
        cache_size=CACHE_SIZE,
    )

    writer.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("Training done!")


if __name__ == '__main__':
    print("Start training SimAM with true 2D input in PyTorch (DVH loss removed, WL/WR balanced loading enabled)")
    main()
    print("Training done!")