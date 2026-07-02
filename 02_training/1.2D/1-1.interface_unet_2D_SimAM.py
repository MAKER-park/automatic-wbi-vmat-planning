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


##################################################################################
# 추론(예측) 후 결과 저장하는 함수 (2.5D 슬라이스 단위)
##################################################################################
def predict_2D_and_save(
    file_name, model, data_4d, output_dir, 
    channel_weights=None, device='cuda'
):
    """
    file_name    : 결과 파일명에 사용할 식별자 (ex: "Patient01_CT_Contour.npy")
    model        : load_state_dict된 U-Net2D_SimAM
    data_4d      : [H, W, Z, C] 형태의 4D numpy (CT+Contour 등)
    output_dir   : 예측 결과를 저장할 폴더 경로
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

    # 데이터 들어온거 확인
    print(f"Data shape: {data_4d.shape} (H, W, Z, C)")

    pred_volume = np.zeros((H, W, Z), dtype=np.float32) 

    # slice loop
    for z in tqdm(range(Z), desc=f"Predicting {file_name}"):
        # (H, W, C) 짜리 2D 입력 구성
        input_2D = data_4d[:, :, z, :]  # (H, W, C)
        
        # 채널 가중치 적용
        for c in range(C):
            input_2D[:, :, c] *= channel_weights[c]

        # 차원 변환 => (1, 채널, H, W)
        input_2D = np.transpose(input_2D, (2,0,1))  # (C, H, W)
        input_2D = input_2D[None, ...]             # (1, C, H, W)

        # 텐서로 변환
        input_t = torch.tensor(input_2D, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            pred_t = model(input_t)          # (1, 1, H, W)
            pred_np = pred_t.cpu().numpy()[0, 0]  # (H, W)
            pred_volume[:, :, z] = pred_np

    
    #예측 결과 저장
    base_name = os.path.splitext(file_name)[0]
    save_name = f"{base_name}_pred_dose.npy"
    save_path = os.path.join(output_dir, save_name)
    np.save(save_path, pred_volume)
    print(f"[Saved] {save_path} shape={pred_volume.shape}")

    # 예측 결과 저장할 배열
    # pred_volume = np.zeros((H, W, Z), dtype=np.float32)
    
    # # 슬라이스 별로 예측
    # for z in tqdm(range(Z), desc=f"Predict {file_name}"):
    #     # (H, W, C) 짜리 2D 입력 구성
    #     input_2D = make_2D_input_volume(data_4d, z, channel_weights)
        
    #     # 차원 변환 => (1, 채널, H, W)
    #     input_2_5D = np.transpose(input_2_5D, (2,0,1))  # (채널, H, W)
    #     input_2_5D = input_2_5D[None, ...]             # (1, 채널, H, W)

    #     input_t = torch.tensor(input_2_5D, dtype=torch.float32, device=device)
        
    #     with torch.no_grad():
    #         pred_t = model(input_t)          # (1, 1, H, W)
    #         pred_np = pred_t.cpu().numpy()[0, 0]  # (H, W)
    #     pred_volume[..., z] = pred_np
    
    # 예측 결과 저장
    # 예: "Patient01_CT_Contour.npy" -> "Patient01_CT_Contour_pred_dose.npy"
    # base_name = os.path.splitext(file_name)[0]
    # save_name = f"{base_name}_pred_dose.npy"
    # save_path = os.path.join(output_dir, save_name)
    # np.save(save_path, pred_volume)
    # print(f"[Saved] {save_path} shape={pred_volume.shape}")

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
    original_channel_count = 9  #  9
    in_channels = original_channel_count  # 24
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
    test_ct_contour_dir = "../final_dataset/Test/CT_and_Contour"  # 예: 테스트 세트 폴더
    output_dir = "prediction_2D_npy"                  # 예: 예측 결과 저장 폴더
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
        predict_2D_and_save(
            file_name, model, data_4d, output_dir,
            channel_weights=channel_weights,
            device=device
        )

if __name__ == '__main__':
    main()
