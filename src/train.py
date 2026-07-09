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
    scheduler,
    device,
    epochs,
    run_dir,
    checkpoint=None,
    physics_loss=True,
):
    train_losses = []
    valid_losses = []

    train_data_losses = []
    valid_data_losses = []

    train_physics_losses = []
    valid_physics_losses = []

    start_epoch = 0

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "scheduler_state_dict" in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        start_epoch = checkpoint["epoch"] + 1

        train_losses = checkpoint["train_losses"].copy()
        valid_losses = checkpoint["valid_losses"].copy()

        train_data_losses = checkpoint.get("train_data_losses", []).copy()
        valid_data_losses = checkpoint.get("valid_data_losses", []).copy()

        train_physics_losses = checkpoint.get("train_physics_losses", []).copy()
        valid_physics_losses = checkpoint.get("valid_physics_losses", []).copy()

    for epoch in range(start_epoch, epochs):
        model.train()

        train_loss = 0.0
        train_data_loss = 0.0
        train_physics_loss = 0.0
        train_samples = 0

        for input, target in tqdm(train_dataloader):
            batch_size = input.shape[0]

            input = input.to(device)
            input.requires_grad_(True)
            target = target.to(device)

            pred = model(input)

            optimizer.zero_grad()

            total_loss, data_loss, phys_loss = criterion(input, pred, target)

            if physics_loss:
                loss = total_loss
            else:
                loss = data_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size
            train_data_loss += data_loss.item() * batch_size
            train_physics_loss += phys_loss.item() * batch_size
            train_samples += batch_size

        train_loss /= train_samples
        train_data_loss /= train_samples
        train_physics_loss /= train_samples

        model.eval()

        valid_loss = 0.0
        valid_data_loss = 0.0
        valid_physics_loss = 0.0
        valid_samples = 0

        for input, target in valid_dataloader:
            batch_size = input.shape[0]
            
            input = input.to(device)
            input.requires_grad_(True)
            target = target.to(device)

            pred = model(input)

            total_loss, data_loss, phys_loss = criterion(input, pred, target)

            if physics_loss:
                loss = total_loss
            else:
                loss = data_loss

            valid_loss += loss.item() * batch_size
            valid_data_loss += data_loss.item() * batch_size
            valid_physics_loss += phys_loss.item() * batch_size
            valid_samples += batch_size

        valid_loss /= valid_samples
        valid_data_loss /= valid_samples
        valid_physics_loss /= valid_samples

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        train_data_losses.append(train_data_loss)
        valid_data_losses.append(valid_data_loss)

        train_physics_losses.append(train_physics_loss)
        valid_physics_losses.append(valid_physics_loss)

        print(
            f"Epoch {epoch}: "
            f"train={train_loss:.6f} "
            f"(data={train_data_loss:.6f}, physics={train_physics_loss:.6f}) | "
            f"valid={valid_loss:.6f} "
            f"(data={valid_data_loss:.6f}, physics={valid_physics_loss:.6f})"
        )
        
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "train_losses": train_losses,
                "valid_losses": valid_losses,
                "train_data_losses": train_data_losses,
                "valid_data_losses": valid_data_losses,
                "train_physics_losses": train_physics_losses,
                "valid_physics_losses": valid_physics_losses,
            },
            checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth",
        )

        if scheduler is not None:
            scheduler.step(valid_loss)

    return {
        "train": train_losses,
        "valid": valid_losses,
        "train_data": train_data_losses,
        "valid_data": valid_data_losses,
        "train_physics": train_physics_losses,
        "valid_physics": valid_physics_losses,
    }