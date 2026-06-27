import torch
import torch.nn as nn

from tqdm import tqdm

from pathlib import Path
from datetime import datetime


def train_model(
    model,
    train_dataloader,
    valid_dataloader,
    criterion,
    optimizer,
    device,
    epochs,
    run_dir,
    checkpoint=None,
    physics_loss=True,
):
    train_losses = []
    valid_losses = []

    start_epoch = 0

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

        train_losses = checkpoint["train_losses"].copy()
        valid_losses = checkpoint["valid_losses"].copy()

    for epoch in range(start_epoch, epochs):
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

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": train_losses,
                "valid_losses": valid_losses,
            },
            run_dir / "checkpoints" / f"checkpoint_epoch_{epoch + 1}.pth",
        )
        
    return train_losses, valid_losses
