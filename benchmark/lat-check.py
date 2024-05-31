import argparse
from datetime import datetime

import torch
from torch import nn
from torch import optim
from PIL import Image

from utils.dataloader import infer_test_prep
import config


# Model prep
def model_setup(path):
    model = torch.load(path).eval()
    model.to(config.DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return model, loss_fn, optimizer


# Inference
def inference(model, transform, image_files):
    start = datetime.now()
    with torch.no_grad():
        for image_file in image_files:
            image = Image.open(image_file)
            image = transform(image).unsqueeze(0)  # Add batch dimension
            image = image.to(config.DEVICE)

            _ = model(image)
    end = datetime.now()

    # calculate total time
    time_delta = round((end - start).total_seconds(), 2)
    print(f'Total inference time\t: {time_delta} seconds')
    print(f'Latency per image\t: {round(time_delta / 100, 2)} seconds')


def main(args):
    transform, _, infer_images = infer_test_prep(args.model)
    model, _, _ = model_setup(args.path)

    inference(model, transform, infer_images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(config.MODEL_WEIGHTS_MAP.keys()), required=True, help="DL model to run")
    parser.add_argument('--path')

    args = parser.parse_args()
    print(args)

    main(args)
