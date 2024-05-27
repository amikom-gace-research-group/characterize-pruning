import torch
from torchvision import models


# Execution
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model configs
MODEL_WEIGHTS_MAP = {
    'mobilenet_v3_small': models.MobileNet_V3_Small_Weights,
    'efficientnet_v2_l': models.EfficientNet_V2_L_Weights,
    'swin_v2_s': models.Swin_V2_S_Weights,
    'swin_v2_b': models.Swin_V2_B_Weights,
}

# TQDM config
BAR_FORMAT='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
