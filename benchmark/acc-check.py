import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn
from torch import optim
from tqdm import tqdm


# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Data prep
transform = models.EfficientNet_V2_L_Weights.DEFAULT.transforms()
testset = torchvision.datasets.Flowers102(root='./data', split="test", download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# Model prep
model = models.efficientnet_v2_l()
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 102)
state_dict = torch.load('./models/pruned-effv2l-flowers-1.pth', map_location=torch.device('cuda'))
model.load_state_dict(state_dict)
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(tqdm(dataloader)):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


test(testloader, model, loss_fn)