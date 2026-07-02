#!/bin/bash

# conda 환경 초기화
source ~/anaconda3/etc/profile.d/conda.sh
conda activate 11.8

# 파일 삭제 및 실행
rm -rf *.png *.pth gradcam Model_Weight_2D_SimAM* epoch_image pred*
python train.py
# python interface_unet_2.5D_SimAM.py
