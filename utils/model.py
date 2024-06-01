import pathlib
import re

import torch
from torch import nn, optim
import torchvision.models as models

import config


def setup_model(model_type):
    weights = config.MODEL_WEIGHTS_MAP[model_type].DEFAULT
    model_class = getattr(models, model_type)
    model = model_class(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    # Filter
    re_densenet = re.compile(r"^densenet.*$")
    re_vit = re.compile(r"^vit.*$")

    if re_densenet.match(model_type): 
        model.classifier = nn.Linear(model.classifier.in_features, 102)
    if re_vit.match(model_type):
        model.heads[-1] = nn.Linear(model.heads[-1].in_features, 102)
    else:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 102)  # type: ignore

    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return model, loss_fn, optimizer


def save_model(model, save_as, path):
    def prepare_path(path: str):
        split_path = path.split("/")
        filename = split_path[-1]
        folder_path = "/".join(split_path[:-1])

        base = pathlib.Path(folder_path)
        base.mkdir(parents=True, exist_ok=True)

        return base / filename

    torch.save(
        model.state_dict() if save_as == 'state-dict' else model,
        prepare_path(path))
