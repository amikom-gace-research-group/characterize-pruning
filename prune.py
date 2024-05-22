import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn
from torch import optim
from tqdm import tqdm
from torch.nn.utils import prune


# Model structure
model = models.efficientnet_v2_l()
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 102)
state_dict = torch.load('./models/effv2l-flowers-1.pth', map_location=torch.device('cuda'))
model.load_state_dict(state_dict)
print('model loaded!')


# Prune
# Identify the parameters to prune
parameters_to_prune = []
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        parameters_to_prune.append((module, 'weight'))

# Apply global unstructured pruning
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.5,  # Pruning 50% of the parameters globally
)

# Remove the reparameterization
for module, param_name in parameters_to_prune:
    prune.remove(module, param_name)


# Verify the pruning
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

print('model pruned!')
print(f'Total sparsity: {100 * (1 - total_nonzero_weights / total_weights):.2f}%')


# Save only the state dict
torch.save(model.state_dict(), './models/pruned-effv2l-flowers-1.pth')
# Save state dict and model structure
# torch.save(model, 'pruned_vgg16_entire_model.pth')