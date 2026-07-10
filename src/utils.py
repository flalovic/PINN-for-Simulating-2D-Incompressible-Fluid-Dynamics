import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn


def regions_overlap(r1: dict, r2: dict) -> bool:
    x_overlap = r1['x_min'] < r2['x_max'] and r2['x_min'] < r1['x_max']
    y_overlap = r1['y_min'] < r2['y_max'] and r2['y_min'] < r1['y_max']
    return x_overlap and y_overlap

def check_regions(regions: list[dict]) -> bool:
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            if regions_overlap(regions[i], regions[j]):
                return False
    return True

def knn_indices(query: torch.Tensor, source: torch.Tensor, k: int) -> torch.Tensor:
    """
    Za svaku query tacku vraca indekse k najblizih source tacaka.
    query: (Q, 2), source: (S, 2) -> idx: (Q, k).
    Selekcija (topk) je nediferencijabilna po dizajnu; gradijent tece kroz
    naknadno racunanje relativnih pozicija i gather-ovanih feature-a.

    NB: koristi torch.cdist (O(Q*S) memorije). Za velike point-cloud-ove po stanju
    razmotri torch_cluster.radius/knn.
    """
    with torch.no_grad():
        dist = torch.cdist(query, source)          # (Q, S)
        k = min(k, source.shape[0])
        idx = torch.topk(dist, k, dim=1, largest=False).indices  # (Q, k)
    return idx


def make_uniform_grid(node_pos: torch.Tensor, size: int) -> torch.Tensor:
    """
    Gradi uniformni size x size grid koji pokriva bounding box datih tacaka.
    Vraca (size*size, 2). Konstantan tenzor (bez gradijenta ka ulazu).
    """
    mins = node_pos.min(dim=0).values
    maxs = node_pos.max(dim=0).values
    xs = torch.linspace(mins[0].item(), maxs[0].item(), size, device=node_pos.device)
    ys = torch.linspace(mins[1].item(), maxs[1].item(), size, device=node_pos.device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)  # (size*size, 2)


def mlp(dims, act=nn.GELU, last_act=False):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last_act:
            layers.append(act())
    return nn.Sequential(*layers)

def normalize_data(df: pd.DataFrame, mean, std) -> pd.DataFrame:
    """
    Normalizujemo podatke u DF-u koristeci Z-score normalizaciju.
    """
    return (df - mean) / std

def compute_metrics(y_true, y_pred):
    """
    Računa osnovne metrike za regresiju.

    Args:
        y_true: stvarne vrednosti (numpy array ili list)
        y_pred: predviđene vrednosti

    Returns:
        dict: MAE, MSE, RMSE, R2, MaxAbsError, RelL2
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shapes do not match: y_true has shape {y_true.shape}, "
            f"but y_pred has shape {y_pred.shape}"
        )

    diff = y_pred - y_true
    abs_diff = np.abs(diff)
    sq_diff = diff ** 2

    mae = float(np.mean(abs_diff))
    mse = float(np.mean(sq_diff))
    rmse = float(np.sqrt(mse))
    max_abs_error = float(np.max(abs_diff))

    ss_res = float(np.sum(sq_diff))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    denom = float(np.sum(y_true ** 2))
    rel_l2 = float(np.sqrt(ss_res / denom)) if denom > 0 else float("nan")

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MaxAbsError": max_abs_error,
        "RelL2": rel_l2,
    }


def evaluate_model(
    model,
    df: pd.DataFrame,
    input_col_names,
    target_col_names,
    mean,
    std,
    device,
    return_predictions=False,
):
    """
    Evaluira model na datom DataFrame-u i računa metrike na originalnoj skali.

    Args:
        model: PyTorch model
        df: DataFrame sa podacima
        input_col_names: lista kolona za ulaze
        target_col_names: lista kolona za ciljeve
        mean: pandas Series/DataFrame sa sredinama za sve kolone
        std: pandas Series/DataFrame sa standardnim devijacijama
        device: torch device
        return_predictions: ako je True, vraća i predviđene vrednosti

    Returns:
        metrics_dict: dictionary sa svim metrikama
        optional pred_df: DataFrame sa predviđenim vrednostima (ako je return_predictions=True)
    """
    model.eval()

    input_values = df[input_col_names].to_numpy(dtype=np.float32)
    target_values = df[target_col_names].to_numpy(dtype=np.float32)

    input_mean = mean[input_col_names].to_numpy(dtype=np.float32)
    input_std = std[input_col_names].to_numpy(dtype=np.float32)
    target_mean = mean[target_col_names].to_numpy(dtype=np.float32)
    target_std = std[target_col_names].to_numpy(dtype=np.float32)

    input_norm = (input_values - input_mean) / input_std
    input_tensor = torch.tensor(input_norm, dtype=torch.float32, device=device)

    with torch.no_grad():
        pred_norm = model(input_tensor).cpu().numpy()

    pred_values = pred_norm * target_std + target_mean
    true_values = target_values

    metrics = {
        "all": compute_metrics(true_values, pred_values)
    }

    for i, col in enumerate(target_col_names):
        metrics[col] = compute_metrics(
            true_values[:, i],
            pred_values[:, i],
        )

    if return_predictions:
        return metrics, pd.DataFrame(pred_values, columns=target_col_names)

    return metrics

def flatten_metrics(metrics, components=("all", "U_x", "U_y", "p")):
    """
    Poravna ugnjezdjen izlaz `evaluate_model`-a u ravan dict pogodan za red CSV-a.
    Ključevi su '{metrika}_{komponenta}', npr. 'R2_all', 'MAE_U_x', 'RMSE_p'.
    Uključuje sve metrike koje vraća `compute_metrics`
    (MAE, MSE, RMSE, R2, MaxAbsError, RelL2).
    """
    flat = {}
    for comp in components:
        if comp not in metrics:
            continue
        for name, value in metrics[comp].items():
            flat[f"{name}_{comp}"] = value
    return flat


def split_by_box(df, box):
    """
    Dijeli DataFrame na tačke UNUTAR i IZVAN pravougaonog (x,y) regiona `box`.
    box: dict sa 'x_min','x_max','y_min','y_max'.
    Vraća (df_inside, df_outside). Granica se broji kao unutra (>=, <=).
    """
    inside_mask = (
        (df['x'] >= box['x_min']) & (df['x'] <= box['x_max']) &
        (df['y'] >= box['y_min']) & (df['y'] <= box['y_max'])
    )
    return df[inside_mask].copy(), df[~inside_mask].copy()


def get_domain_ranges(df, input_col_names, overrides=None):
    """
    Fizičke granice domena po ulaznoj koloni, iz podataka (min, max).
    `overrides` (npr. {'x': (0.0, 1.0), 'y': (0.0, 1.0)}) gaze izvedene granice -
    korisno da se kolokacija fiksira na tačan geometrijski domen umjesto na min/max uzoraka.
    """
    ranges = {c: (float(df[c].min()), float(df[c].max())) for c in input_col_names}
    if overrides:
        ranges.update({c: tuple(v) for c, v in overrides.items()})
    return ranges


def sample_collocation(n, input_col_names, ranges, mean, std, device, re_values=None, generator=None):
    """
    Uzorkuje `n` KOLOKACIONIH tačaka (bez labela) uniformno po domenu i vraća ih
    NORMALIZOVANE (Z-score, kao i ulaz modela), sa requires_grad=True za autograd.

    ranges: dict kolona -> (low, high) u FIZIČKIM jedinicama (v. get_domain_ranges).
    Vraća tenzor oblika (n, len(input_col_names)).
    """
    lows = torch.tensor([ranges[c][0] for c in input_col_names], dtype=torch.float32)
    highs = torch.tensor([ranges[c][1] for c in input_col_names], dtype=torch.float32)
    m = torch.tensor([float(mean[c]) for c in input_col_names], dtype=torch.float32)
    s = torch.tensor([float(std[c]) for c in input_col_names], dtype=torch.float32)

    unit = torch.rand(n, len(input_col_names), generator=generator)
    phys = lows + unit * (highs - lows)            

    # Ne zelimo da mreza uci na Re vrijedostima koje nisu u TRAIN skupu
    if re_values is not None:
        re_idx = input_col_names.index("re")

        re_values = torch.tensor(re_values, dtype=torch.float32)

        indices = torch.randint(len(re_values), (n,), generator=generator)

        phys[:, re_idx] = re_values[indices]

    norm = (phys - m) / s                           

    return norm.to(device).requires_grad_(True)


def create_run_directory(frac_size=None, label=None):
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if frac_size is not None:
        run_name += f"_frac{int(100 * frac_size)}"
    if label is not None:
        run_name += f"_{label}"

    run_dir = Path("runs") / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    return run_dir


def plot_training_history(history_df, output_path=None, title=None):
    def plot(df, axes):
        for ax in axes:
            ax.ticklabel_format(
                axis="y",
                style="sci",
                scilimits=(-2, 2),
                useMathText=True,
            )
            ax.grid(True)

        df[["train", "valid"]].plot(ax=axes[0])
        axes[0].set_title("Total loss")
        axes[0].set_xlabel("Epoch")

        df[["train_data", "valid_data"]].plot(ax=axes[1])
        axes[1].set_title("Data loss")
        axes[1].set_xlabel("Epoch")

        df[["train_physics", "valid_physics"]].plot(ax=axes[2])
        axes[2].set_title("Physics loss")
        axes[2].set_xlabel("Epoch")

    n_epochs = len(history_df)

    first_end = min(100, n_epochs)
    middle_start = first_end
    middle_end = max(middle_start, n_epochs - 100)

    fig, ax = plt.subplots(4, 3, figsize=(12, 12), tight_layout=True)

    plot(history_df, ax[0])
    plot(history_df.iloc[:first_end], ax[1])
    plot(history_df.iloc[middle_start:middle_end], ax[2])
    plot(history_df.iloc[middle_end:], ax[3])

    if title is not None:
        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path is not None:
        fig.savefig(output_path, dpi=300)

    return fig, ax