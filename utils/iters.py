import torch
from tqdm import tqdm

import config


def train(dataloader, model, loss_fn, optimizer, epoch, num_epochs):
    model.train()

    # for progress bar
    bar = tqdm(dataloader, desc=f"{epoch+1}/{num_epochs}", dynamic_ncols=True, bar_format=config.BAR_FORMAT)
    running_loss = 0.0

    for batch, (X, y) in enumerate(bar):
        X, y = X.to(config.DEVICE), y.to(config.DEVICE)

        # Compute prediction error
        pred = model(X)  # This is calling the forward() function in the model
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()  # gradients are computed here
        optimizer.step()  # weights and biases are updated, using the gradients from loss.backward()
        optimizer.zero_grad()  # reset gradients to zero

        # Update running loss
        running_loss += loss.item()
        current_loss = running_loss / (batch + 1)

        # Update progress bar with info
        bar.set_postfix(loss="{:.3f}".format(current_loss))


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss, correct = 0, 0
    bar = tqdm(dataloader, desc=f"Eval", dynamic_ncols=True, bar_format=config.BAR_FORMAT)

    with torch.no_grad():
        for batch, (X, y) in enumerate(bar):
            X, y = X.to(config.DEVICE), y.to(config.DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")