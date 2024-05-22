import argparse

import torch
import torchvision.models as models
from torch import nn
from torch import optim

import config
from utils import dataloader, iters


def model_setup(model_type):
    weights = config.MODEL_WEIGHTS_MAP[model_type].DEFAULT
    model_class = getattr(models, model_type)
    model = model_class(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 100)

    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # set model to appropriate device
    model.to(config.DEVICE)

    return model, loss_fn, optimizer


def main(args):
    trainloader, testloader = dataloader.data_prep(args.model, args.dataset, args.batch_size)
    model, loss_fn, optimizer = model_setup(args.model)

    epochs = args.epochs
    for t in range(epochs):
        iters.train(trainloader, model, loss_fn, optimizer, t, epochs)

    print()
    iters.test(testloader, model, loss_fn)

    torch.save(
        model.state_dict() if args.save_as == 'state-dict' else model,
        args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['vgg16', 'efficientnet_v2_l'], required=True, help="DL model to run")
    parser.add_argument('--dataset', choices=['CIFAR100', 'Flowers102'], default="CIFAR100", required=True, help="Dataset to train the model")
    parser.add_argument('--epochs', default=5, type=int, help="Number of iterations/epochs for training")
    parser.add_argument('--save-path', required=True, help="Where to save the trained model")
    parser.add_argument('--save-as', choices=['state-dict', 'full'], default='full', help="State-dict will save only weights, full with save the model structure as well")
    parser.add_argument('--batch-size', default=32, type=int, help="How many images per batch for both train and test set")

    args = parser.parse_args()
    print(args)
    
    main(args)