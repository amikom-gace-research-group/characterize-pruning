import torch
from torchvision import models, datasets


# Execution
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model configs
MODEL_WEIGHTS_MAP = {
    'vgg16': models.VGG16_Weights,
    'efficientnet_v2_l': models.EfficientNet_V2_L_Weights,
}

# TQDM config
BAR_FORMAT='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'