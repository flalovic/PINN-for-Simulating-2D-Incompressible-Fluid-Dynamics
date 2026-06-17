import torch
import torch.nn as nn

from tqdm import tqdm


def train_model(model, train_dataloader, valid_dataloader, criterion, optimizer, device, EPOCHS, physics_loss: bool = True):
    train_losses = []
    valid_losses = []
    for epoch in range(EPOCHS):
        model.train()

        train_loss = 0.0

        for input, target in tqdm(train_dataloader):

            input = input.to(device)
            input.requires_grad_(True)
            target = target.to(device)

            pred = model(input)

            optimizer.zero_grad()
            if(physics_loss):
                loss = criterion(input, pred, target)
            else:
                loss = criterion(pred, target)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        model.eval()
        valid_loss = 0.0

        for input, target in valid_dataloader:
            input = input.to(device)
            input.requires_grad_(True)
            target = target.to(device)

            pred = model(input)
            if(physics_loss):
                loss = criterion(input, pred, target)
            else:
                loss = criterion(pred, target)
            valid_loss += loss.item()

        valid_loss /= len(valid_dataloader)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print(f"Epoch {epoch}: train loss: {train_loss}, valid loss: {valid_loss}")
    return train_losses, valid_losses
