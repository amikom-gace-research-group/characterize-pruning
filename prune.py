import argparse
from typing import Union, Tuple

import torch
from torch import nn
from torch.nn.utils import prune


LAYER_KINDS = {
    "Linear": (nn.Linear),
    "Conv2d": (nn.Conv2d),
    "Complete": (nn.Linear, nn.Conv2d) 
}

PRUNING_METHODS = {
    "L1Unstructured": prune.L1Unstructured,
    "RandomUnstructured": prune.RandomUnstructured
}


def check_pruning_percentage(model: nn.Module):
    def count_nonzero_weights(module):
        return torch.sum(module.weight != 0).item()

    total_nonzero_weights = 0
    total_weights = 0

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nonzero = count_nonzero_weights(module)
            total_nonzero_weights += nonzero
            total_weights += module.weight.nelement()
            print(f'{name}: {nonzero}/{module.weight.nelement()}')

    print(f'Total sparsity: {100 * (1 - total_nonzero_weights / total_weights):.2f}%')


def prune_model(
    model: nn.Module,
    layer_kind: tuple[nn.Module],
    pruning_method: prune.BasePruningMethod,
    amount=0.5,
    global_unstructured=False
):
    """
    Prune the model with the pruning method, and amount of sparsity of your choice

    Args:
        - model: The loaded model you want to prune
        - layer_kind: What layer to prune 
        - pruning_method: What pruning_method to use
        - amount (float): How much of the layer (or network) will be zeroed. 0.5 is equal to 50%
        - global_unstructured (bool): set to True to use global_unstructured pruning
    """
    parameters_to_prune = []
    for _, module in model.named_modules():
        if isinstance(module, layer_kind):  # type: ignore
            parameters_to_prune.append((module, 'weight'))
            
    # Apply global unstructured pruning
    if global_unstructured:
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=pruning_method,  
            amount=amount, 
        )

    # Remove the reparameterization
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)

    return model


def main(args):
    model_to_prune: nn.Module = torch.load(args.path)
    layer_kind = LAYER_KINDS[args.layer_kind]
    pruning_method = PRUNING_METHODS[args.pruning_method]

    pruned_model = prune_model(
        model=model_to_prune,
        layer_kind=layer_kind,
        pruning_method=pruning_method,
        amount=args.amount,
        global_unstructured=args.global_unstructured
    )

    if args.check_percentage:
        check_pruning_percentage(pruned_model)

    print("DONE!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--layer-kind", choices=list(LAYER_KINDS.keys()))
    parser.add_argument("--pruning-method", choices=list(PRUNING_METHODS.keys()))
    parser.add_argument("--amount", type=float)
    parser.add_argument("--global-unstructured", action="store_true")
    parser.add_argument("--check-percentage", action="store_true")

    args = parser.parse_args()
    print(args)
    
    main(args)
