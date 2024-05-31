import argparse

import torch
from torch import nn

import config
from utils import dataloader, iters


def load_model(path: str):
    model = torch.load(path).eval()
    model.to(config.DEVICE)

    loss_fn = nn.CrossEntropyLoss()

    return model, loss_fn


def main(args):
    _, testloader = dataloader.data_prep(args.model, args.dataset, args.batch_size)
    model, loss_fn = load_model(args.path)

    iters.silent_test(testloader, model, loss_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(config.MODEL_WEIGHTS_MAP.keys()), required=True, help="DL model to run")
    parser.add_argument('--dataset', choices=['CIFAR100', 'Flowers102'], default="CIFAR100", required=True, help="Dataset to train the model")
    parser.add_argument('--path', required=True, help="Where to save the trained model")
    parser.add_argument('--batch-size', default=32, type=int, help="How many images per batch for both train and test set")

    args = parser.parse_args()
    print(args)

    main(args)
