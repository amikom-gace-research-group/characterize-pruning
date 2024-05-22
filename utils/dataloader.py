import torch
from torchvision import datasets

import config


def data_prep(model_type, dataset_str, batch_size):
    transform = config.MODEL_WEIGHTS_MAP[model_type].DEFAULT.transforms()

    dataset = getattr(datasets, dataset_str)

    trainset = dataset(root='./data', split="train", download=True, transform=transform)
    testset = dataset(root='./data', split="test", download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader