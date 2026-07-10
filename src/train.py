import torch
import torch.nn as nn

from tqdm import tqdm

from pathlib import Path
from datetime import datetime

from src.utils import sample_collocation


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


def train_collocation(
    model,
    train_dataloader,
    valid_dataloader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    run_dir,
    input_col_names,
    ranges,
    mean,
    std,
    n_collocation=4096,
):
    """
    PINN trening sa KOLOKACIJOM: data loss se računa na labeliranim tačkama, a fizički
    rezidual na `n_collocation` tačaka uniformno uzorkovanih po domenu
    Time fizika ograničava rješenje i tamo gdje nema obilježenih podataka.

    Kompatibilno sa istim vizuelizacijama kao `train_model` (isti ključevi u history-ju) i
    sličan princip rada.
    
    Model se bira po najboljem validacionom DATA gubitku (`best_model.pth`).
    """
    history = {k: [] for k in
               ["train", "valid", "train_data", "valid_data", "train_physics", "valid_physics"]}
    c_phys = criterion.c_physics
    best_valid_data = float("inf")

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        tr_total = tr_data = tr_phys = 0.0
        tr_samples = tr_steps = 0

        for input, target in tqdm(train_dataloader):
            input = input.to(device)
            target = target.to(device)
            batch_size = input.shape[0]

            optimizer.zero_grad()

            # data loss na labeliranim tačkama
            pred = model(input)
            data_loss = criterion.mse(pred, target)

            # fizika na kolokacionim (neobilježenim) tačkama
            coll = sample_collocation(n_collocation, input_col_names, ranges, mean, std, device)
            phys_loss = criterion.physics_loss(coll, model(coll))

            loss = data_loss + c_phys * phys_loss
            loss.backward()
            optimizer.step()

            tr_total += loss.item() * batch_size
            tr_data += data_loss.item() * batch_size
            tr_samples += batch_size
            tr_phys += phys_loss.item()
            tr_steps += 1

        train_total = tr_total / tr_samples
        train_data = tr_data / tr_samples
        train_phys = tr_phys / tr_steps

        # ---- validacija: data loss na cijelom valid skupu, fizika na kolokaciji ----
        model.eval()
        va_data = 0.0
        va_samples = 0
        with torch.no_grad():
            for input, target in valid_dataloader:
                input = input.to(device)
                target = target.to(device)
                va_data += criterion.mse(model(input), target).item() * input.shape[0]
                va_samples += input.shape[0]
        valid_data = va_data / va_samples

        coll = sample_collocation(n_collocation, input_col_names, ranges, mean, std, device)
        valid_phys = criterion.physics_loss(coll, model(coll)).item()
        valid_total = valid_data + c_phys * valid_phys

        history["train"].append(train_total)
        history["valid"].append(valid_total)
        history["train_data"].append(train_data)
        history["valid_data"].append(valid_data)
        history["train_physics"].append(train_phys)
        history["valid_physics"].append(valid_phys)

        print(
            f"Epoch {epoch}: "
            f"train={train_total:.6f} (data={train_data:.6f}, physics={train_phys:.6f}) | "
            f"valid={valid_total:.6f} (data={valid_data:.6f}, physics={valid_phys:.6f})"
        )

        # najbolji model po validacionom DATA gubitku
        if valid_data < best_valid_data:
            best_valid_data = valid_data
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "valid_data": valid_data},
                run_dir / "best_model.pth",
            )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": (
                    scheduler.state_dict() if scheduler is not None else None
                ),
                "history": history,
                "best_valid_data": best_valid_data,
            },
            checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth",
        )

        if scheduler is not None:
            scheduler.step(valid_data)

    return history