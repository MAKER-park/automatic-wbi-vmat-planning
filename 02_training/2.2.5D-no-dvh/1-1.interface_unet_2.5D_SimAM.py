import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

##################################################################################
# 전처리에 사용할 클리핑 범위(학습 때와 동일하게)
##################################################################################
clip_min = -300
clip_max = 800

#######################################
# SimAM Layer
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

#######################################################################################

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

#######################################
# U-Net with SimAM
#######################################
class UNet2D_SimAM(nn.Module):
    def __init__(self, in_channels=9, out_channels=1):
        super(UNet2D_SimAM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # down
        self.conv1 = conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = conv_block(128, 128)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = conv_block(128, 256)

        # up
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
        c1 = self.conv1(x)  # -> 32
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)  # -> 64
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)  # -> 128
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)  # -> 128
        p4 = self.pool4(c4)

        c5 = self.conv5(p4)  # -> 256

        x6 = self.up6_convT(c5)  # 256->128
        x6 = torch.cat([x6, c4], dim=1)  # 128+128=256
        x6 = self.up6_block(x6)  # ->128

        x7 = self.up7_convT(x6)  # 128->128
        x7 = torch.cat([x7, c3], dim=1)  # 128+128=256
        x7 = self.up7_block(x7)  # ->128

        x8 = self.up8_convT(x7)  # 128->64
        x8 = torch.cat([x8, c2], dim=1)  # 64+64=128
        x8 = self.up8_block(x8)  # ->64

        x9 = self.up9_convT(x8)  # 64->32
        x9 = torch.cat([x9, c1], dim=1)  # 32+32=64
        x9 = self.up9_block(x9)  # ->32

        out = self.final_conv(x9)  # 32->1
        out = self.sigmoid(out)
        return out


##################################################################################
# 2.5D 입력 데이터를 만드는 함수 (추론용)
##################################################################################
def make_2_5D_input_volume(data_4d, z, slice_window, channel_weights):
    """
    data_4d: [H, W, Z, C] 형태의 4D numpy 배열
    z      : 현재 예측할 슬라이스 인덱스
    slice_window: 주변 슬라이스 몇 개 포함할지
    channel_weights: 학습 때 적용했던 채널 가중치 (길이=C)
    
    return: shape이 [H, W, (2*slice_window+1)*C] 인 numpy
    """
    H, W, Z, C = data_4d.shape
    
    slices_ = []
    for offset in range(-slice_window, slice_window + 1):
        z_idx = z + offset
        # 경계 처리 (슬라이스가 0~Z-1 범위를 넘어가지 않게)
        if z_idx < 0:
            z_idx = 0
        if z_idx > Z - 1:
            z_idx = Z - 1

        slice_data = data_4d[:, :, z_idx, :].copy()  # shape [H, W, C]
        # 채널 가중치 적용
        for c_idx in range(C):
            slice_data[:, :, c_idx] *= channel_weights[c_idx]
        slices_.append(slice_data)
    
    # axis=-1(채널 차원)으로 concat -> (H, W, C*(2*slice_window+1))
    concat_slices = np.concatenate(slices_, axis=-1)
    return concat_slices

##################################################################################
# 추론(예측) 후 결과 저장하는 함수 (2.5D 슬라이스 단위)
##################################################################################
def predict_2_5D_and_save(
    file_name, model, data_4d, output_dir, 
    slice_window=1, channel_weights=None, device='cuda'
):
    """
    file_name    : 결과 파일명에 사용할 식별자 (ex: "Patient01_CT_Contour.npy")
    model        : load_state_dict된 U-Net2D_SimAM
    data_4d      : [H, W, Z, C] 형태의 4D numpy (CT+Contour 등)
    output_dir   : 예측 결과를 저장할 폴더 경로
    slice_window : 2.5D를 위해 주변 몇 장을 볼지 (학습 시와 동일)
    channel_weights : 예) [1,1,1,1,1,1,1,1] (학습 시와 동일)
    device       : 'cuda' 또는 'cpu'
    """
    if channel_weights is None:
        # 학습 때 가중치가 없었다면 모두 1로
        C = data_4d.shape[-1]
        channel_weights = [1]*C
    
    # 모델 추론 모드
    model.eval()
    
    H, W, Z, C = data_4d.shape
    # 예측 결과 저장할 배열
    pred_volume = np.zeros((H, W, Z), dtype=np.float32)
    
    # 슬라이스 별로 예측
    for z in tqdm(range(Z), desc=f"Predict {file_name}"):
        # (H, W, (2*slice_window+1)*C) 짜리 2.5D 입력 구성
        input_2_5D = make_2_5D_input_volume(data_4d, z, slice_window, channel_weights)
        
        # 차원 변환 => (1, 채널, H, W)
        input_2_5D = np.transpose(input_2_5D, (2,0,1))  # (채널, H, W)
        input_2_5D = input_2_5D[None, ...]             # (1, 채널, H, W)

        input_t = torch.tensor(input_2_5D, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            pred_t = model(input_t)          # (1, 1, H, W)
            pred_np = pred_t.cpu().numpy()[0, 0]  # (H, W)
        pred_volume[..., z] = pred_np
    
    # 예측 결과 저장
    # 예: "Patient01_CT_Contour.npy" -> "Patient01_CT_Contour_pred_dose.npy"
    base_name = os.path.splitext(file_name)[0]
    save_name = f"{base_name}_pred_dose.npy"
    save_path = os.path.join(output_dir, save_name)
    np.save(save_path, pred_volume)
    print(f"[Saved] {save_path} shape={pred_volume.shape}")

##################################################################################
# 메인 함수: 디렉토리를 순회하며 추론
##################################################################################
def main():
    ############################################################################
    # 1) 디바이스 설정
    ############################################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    ############################################################################
    # 2) 모델 준비 (학습 시 in_channels, out_channels와 동일하게!)
    ############################################################################
    slice_window = 5            # 학습 시 사용한 값과 동일해야 합니다.
    # original_channel_count = 8  # CT+Contour 총 채널 수(예: 8)
    original_channel_count = 9  # 잘못 8로 되어 있음 → 9로 수정
    in_channels = (2*slice_window + 1) * original_channel_count  # 24
    out_channels = 1

    model = UNet2D_SimAM(in_channels=in_channels, out_channels=out_channels).to(device)
    # model = UNet2D_SimAM(in_channels=63, out_channels=1).to(device)


    # 저장된 모델 가중치 경로
    model_weight_path = "./Model_Weight_2D_SimAM/best_model_weight.pth"
    if not os.path.exists(model_weight_path):
        raise FileNotFoundError(f"Model weight not found: {model_weight_path}")

    # 가중치 불러오기
    print("Loading model weights:", model_weight_path)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    print("Model loaded successfully.")

    ############################################################################
    # 3) 추론할 테스트 데이터 디렉토리 & 결과 저장 폴더
    ############################################################################
    test_ct_contour_dir = os.path.join(os.environ.get("WBI_DATA_ROOT", "./data"), "Test", "CT_and_Contour")  # 예: 테스트 세트 폴더
    output_dir = os.path.join(os.environ.get("WBI_OUTPUT_ROOT", "./outputs"), "prediction_npy_2_5D_unet_SimAM")                  # 예: 예측 결과 저장 폴더
    os.makedirs(output_dir, exist_ok=True)

    ############################################################################
    # 4) 채널 가중치 (학습할 때 사용했으면 동일하게 적용)
    ############################################################################
    channel_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1]  # 8채널 예시

    ############################################################################
    # 5) 디렉토리에 있는 .npy 파일들에 대해 추론 수행
    ############################################################################
    npy_files = [f for f in os.listdir(test_ct_contour_dir) if f.endswith('.npy')]
    
    for file_name in npy_files:
        file_path = os.path.join(test_ct_contour_dir, file_name)
        print("="*80)
        print(f"Predicting file: {file_name}")
        
        # (H, W, Z, C) 형태로 로드되었다고 가정
        data_4d = np.load(file_path)  # shape: [H, W, Z, C]
        # 혹시 모를 범위를 학습과 동일하게 clip
        data_4d = np.clip(data_4d, clip_min, clip_max)

        # 추론
        predict_2_5D_and_save(
            file_name, model, data_4d, output_dir,
            slice_window=slice_window,
            channel_weights=channel_weights,
            device=device
        )

if __name__ == '__main__':
    main()
