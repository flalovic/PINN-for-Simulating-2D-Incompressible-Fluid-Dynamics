import numpy as np
import pandas as pd
import torch


def normalize_data(df: pd.DataFrame, mean, std) -> pd.DataFrame:
    """
    Normalizujemo podatke u DF-u koristeci Z-score normalizaciju.
    """
    return (df - mean) / std


def compute_physics_residual_metrics(
    model,
    df: pd.DataFrame,
    input_col_names,
    mean,
    std,
    device,
):
    """
    Računa metrike za PDE rezidualne članove koristeći isti skalirajuci prostor
    kao i trening.

    Vraća:
        dict sa srednjim apsolutnim greškama i RMS vrednostima za:
        - continuity
        - momentum_x
        - momentum_y
    """
    model.eval()

    input_values = df[input_col_names].to_numpy(dtype=np.float32)
    input_mean = mean[input_col_names].to_numpy(dtype=np.float32)
    input_std = std[input_col_names].to_numpy(dtype=np.float32)
    input_norm = (input_values - input_mean) / input_std

    x = torch.tensor(input_norm, dtype=torch.float32, device=device, requires_grad=True)

    with torch.enable_grad():
        pred = model(x)

        u = pred[:, 0]
        v = pred[:, 1]
        p = pred[:, 2]

        u_grad = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]
        v_grad = torch.autograd.grad(
            outputs=v,
            inputs=x,
            grad_outputs=torch.ones_like(v),
            create_graph=True,
            retain_graph=True,
        )[0]
        p_grad = torch.autograd.grad(
            outputs=p,
            inputs=x,
            grad_outputs=torch.ones_like(p),
            create_graph=True,
            retain_graph=True,
        )[0]

        u_x = u_grad[:, 2]
        u_y = u_grad[:, 3]
        v_x = v_grad[:, 2]
        v_y = v_grad[:, 3]
        p_x = p_grad[:, 2]
        p_y = p_grad[:, 3]

        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True,
        )[0][:, 2]
        u_yy = torch.autograd.grad(
            outputs=u_y,
            inputs=x,
            grad_outputs=torch.ones_like(u_y),
            create_graph=True,
            retain_graph=True,
        )[0][:, 3]
        v_xx = torch.autograd.grad(
            outputs=v_x,
            inputs=x,
            grad_outputs=torch.ones_like(v_x),
            create_graph=True,
            retain_graph=True,
        )[0][:, 2]
        v_yy = torch.autograd.grad(
            outputs=v_y,
            inputs=x,
            grad_outputs=torch.ones_like(v_y),
            create_graph=True,
            retain_graph=True,
        )[0][:, 3]

        re = torch.clamp(torch.abs(x[:, 1]) + 1e-8, min=1e-8)

        continuity = u_x + v_y
        momentum_x = u * u_x + v * u_y + p_x - (1.0 / re) * (u_xx + u_yy)
        momentum_y = u * v_x + v * v_y + p_y - (1.0 / re) * (v_xx + v_yy)

    def _stats(tensor):
        tensor = tensor.detach().cpu().numpy()
        return {
            "mean_abs": float(np.mean(np.abs(tensor))),
            "mean_sq": float(np.mean(tensor ** 2)),
            "rmse": float(np.sqrt(np.mean(tensor ** 2))),
            "max_abs": float(np.max(np.abs(tensor))),
        }

    return {
        "continuity": _stats(continuity),
        "momentum_x": _stats(momentum_x),
        "momentum_y": _stats(momentum_y),
    }


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

