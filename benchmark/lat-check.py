import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn
from torch import optim
from tqdm import tqdm
from PIL import Image
import os
from datetime import datetime

# device
device = "cuda" if torch.cuda.is_available() else "cpu"


# data & transforms
transform = models.ViT_B_16_Weights.DEFAULT.transforms()
image_dir = 'assets'
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
warmup_files = image_files[:10]

# Model prep
model = models.vit_()
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 102)
state_dict = torch.load('./models/pruned-effv2l-flowers-1.pth', map_location=torch.device('cuda'))
model.load_state_dict(state_dict)
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Inference
start = datetime.now()
with torch.no_grad():
    for image_file in image_files:
        image = Image.open(image_file)
        image = transform(image).unsqueeze(0)  # Add batch dimension
        image = image.to(device)

        output = model(image)
end = datetime.now()

# calculate total time
time_delta = round((end - start).total_seconds(), 2)
print(f'Total inference time\t: {time_delta} seconds')
print(f'Latency per image\t: {round(time_delta / 100, 2)} seconds')