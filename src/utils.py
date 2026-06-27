import numpy as np
import pandas as pd
import torch


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

