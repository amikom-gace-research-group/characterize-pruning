import argparse
from datetime import datetime

import torch
from torch import nn
from torch import optim
from PIL import Image
from tqdm import tqdm

from utils.dataloader import infer_test_prep
import config


# Model prep
def change_to_dense(model):
    for _, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight = nn.Parameter(module.weight.to_dense())
            
    return model


def model_setup(path):
    model = torch.load(path).eval()
    model.to(config.DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return model, loss_fn, optimizer


# Inference
def inference(model, transform, image_files, progress_bar=False):
    start = datetime.now()
    with torch.no_grad():
        if progress_bar:
            data = tqdm(image_files, dynamic_ncols=True, bar_format=config.BAR_FORMAT)
        else:
            data = image_files

        for _, image_file in enumerate(data):
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

    start_time = datetime.now()
    model, _, _ = model_setup(args.path)
    end_time = datetime.now()
    model_load_delta = round((end_time - start_time).total_seconds(), 2)

    print(f"Model loading takes {model_load_delta}s!")

    start_time = datetime.now()
    model = change_to_dense(model)
    end_time = datetime.now()
    change_to_dense_delta = round((end_time - start_time).total_seconds(), 2)

    print(f"Changing weights to dense takes {change_to_dense_delta}s!")

    inference(model, transform, infer_images, args.progress_bar)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(config.MODEL_WEIGHTS_MAP.keys()), required=True, help="DL model to run")
    parser.add_argument('--path')
    parser.add_argument('--progress-bar', action="store_true")

    args = parser.parse_args()
    print(args)

    main(args)
