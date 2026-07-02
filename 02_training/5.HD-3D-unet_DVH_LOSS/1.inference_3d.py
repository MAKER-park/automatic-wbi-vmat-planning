import os
import numpy as np
import torch
from tqdm import tqdm
import importlib.util
import sys

# Import Model and infer_full_volume from the training script
def import_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

train_script_path = "0.train-HD-3D-Unet-DVH_loss.py"
if not os.path.exists(train_script_path):
    print(f"Error: {train_script_path} not found.")
    sys.exit(1)

train_module = import_from_file("train_script", train_script_path)
Model = train_module.Model
infer_full_volume = train_module.infer_full_volume

# --- Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_CT_DIR = os.path.join(os.environ.get("WBI_DATA_ROOT", "./data"), "Test", "CT_and_Contour")
OUTPUT_DIR = os.path.join(os.environ.get("WBI_OUTPUT_ROOT", "./outputs"), "prediction_npy_3d_hd_unet")
MODEL_PATH = "./Model_Weight_HD_3D_UNet/best_model_weight.pth"
PATCH_SIZE = (32, 256, 256)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 모델 로드
    model = Model(in_ch=9, growth_rate=16, upsample_chan=32, out_ch=1).to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model weight not found at {MODEL_PATH}")
        return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Loaded model weights from {MODEL_PATH}")

    # 2. 테스트 파일 리스트
    if not os.path.exists(TEST_CT_DIR):
        print(f"Error: Test directory {TEST_CT_DIR} not found.")
        return
    test_files = sorted([f for f in os.listdir(TEST_CT_DIR) if f.endswith('.npy')])
    
    # 3. 추론 루프
    for file_name in tqdm(test_files, desc="Inferencing"):
        ct_path = os.path.join(TEST_CT_DIR, file_name)
        
        # infer_full_volume은 (Z, H, W) 결과를 반환함 (0.0 ~ 1.0)
        pred_full_z_first = infer_full_volume(model, ct_path, PATCH_SIZE)
        
        # 레퍼런스 코드와의 호환성을 위해 (H, W, Z)로 변환하여 저장
        # (Z, H, W) -> (H, W, Z)
        pred_full_h_first = np.transpose(pred_full_z_first, (1, 2, 0))
        
        save_name = file_name.replace('.npy', '_pred_dose.npy')
        np.save(os.path.join(OUTPUT_DIR, save_name), pred_full_h_first)

    print(f"Inference complete. Results saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
