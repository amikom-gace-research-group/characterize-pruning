import torch
from torchvision import models


# Execution
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model configs
MODEL_WEIGHTS_MAP = {
    "mobilenet_v3_small": models.MobileNet_V3_Small_Weights,
    "mobilenet_v3_large": models.MobileNet_V3_Large_Weights,
    "efficientnet_b1": models.EfficientNet_B1_Weights,
    "efficientnet_b3": models.EfficientNet_B3_Weights,
    "densenet169": models.DenseNet169_Weights,
    "densenet201": models.DenseNet201_Weights,
    "vit_b_16": models.ViT_B_16_Weights
}

# TQDM config
BAR_FORMAT='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
