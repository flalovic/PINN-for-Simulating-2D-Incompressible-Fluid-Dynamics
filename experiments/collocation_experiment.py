"""
Kolokacioni eksperiment: da li fizika pomaže kada podataka ima MALO?

Trenira dva identična modela na SPARSE skupu (frac_1) i evaluira ih na PUNOM
test skupu (neviđeni Re):

    baseline    : c_physics = 0            (čisto data-driven)
    PINN (coll) : c_physics = w, fizika na KOLOKACIONIM (neobilježenim) tačkama

Ključna razlika u odnosu na raniji sweep: fizički rezidual se sada nameće i tamo
gdje NEMA podataka, što je jedini režim u kojem PINN zaista može da nadmaši data-driven.

Pokretanje (iz korijena repo-a, po mogućnosti na GPU mašini):
    python -m experiments.collocation_experiment --epochs 300 --weights 0 0.1 1.0
"""

import sys
import argparse
import pathlib

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.utils as utils
from src.models import PINN
from src.loss import NavierStokesLoss
from src.dataloader import load_data, gen_dataloaders
from src.train import train_collocation


INPUT_COLS = ["time", "re", "x", "y"]
TARGET_COLS = ["U_x", "U_y", "p"]


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one(c_physics, train_dl, valid_dl, ranges, mean, std, device, epochs, n_coll, run_dir):
    set_seed(42)  # identičan start za svaki model -> fer poređenje
    model = PINN(len(INPUT_COLS), len(TARGET_COLS)).to(device)
    criterion = NavierStokesLoss(c_physics, mean, std)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)

    history = train_collocation(
        model, train_dl, valid_dl, criterion, optimizer, scheduler,
        device, epochs, run_dir, INPUT_COLS, ranges, mean, std, n_collocation=n_coll,
    )
    return model, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", default="frac_1", help="sparse trening podskup (data/real_data/<frac>)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-collocation", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=32768)
    ap.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.1],
                    help="c_physics vrijednosti; 0.0 = data-only baseline")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  frac: {args.frac}  |  epochs: {args.epochs}  |  n_coll: {args.n_collocation}")

    # --- SPARSE trening/validacija ---
    train_df, valid_df, _ = load_data(ROOT / "data" / "real_data" / args.frac, "data")
    mean, std = train_df.mean(), train_df.std()

    tr_n = utils.normalize_data(train_df, mean, std)
    va_n = utils.normalize_data(valid_df, mean, std)

    # --- PUNI test skup (neviđeni Re) za pošten sud ---
    _, _, test_df = load_data(ROOT / "data" / "raw_data", "full_data")

    train_dl, valid_dl, _ = gen_dataloaders(tr_n, va_n, va_n, INPUT_COLS, TARGET_COLS, args.batch_size)

    # kolokacija po TAČNOM geometrijskom domenu (x,y ∈ [0,1]); t,Re iz opsega podataka
    ranges = utils.get_domain_ranges(train_df, INPUT_COLS, overrides={"x": (0.0, 1.0), "y": (0.0, 1.0)})
    print("Kolokacioni domen:", {k: tuple(round(x, 3) for x in v) for k, v in ranges.items()})

    results = []
    for w in args.weights:
        tag = "data_only" if w == 0 else f"coll_cphys_{w}"
        print(f"\n{'='*70}\n{tag}  (c_physics={w})\n{'='*70}")
        run_dir = utils.create_run_directory(label=f"colloc_{args.frac}_{tag}")

        model, _ = run_one(w, train_dl, valid_dl, ranges, mean, std, device,
                           args.epochs, args.n_collocation, run_dir)

        metrics = utils.evaluate_model(model, test_df, INPUT_COLS, TARGET_COLS, mean, std, device)
        row = {"c_physics": w, "tag": tag}
        row.update(utils.flatten_metrics(metrics))   # sve metrike, sve komponente
        results.append(row)

    res = pd.DataFrame(results)
    out = ROOT / "experiments" / f"collocation_results_{args.frac}.csv"
    res.to_csv(out, index=False)

    print(f"\n{'='*70}\nTEST (puni skup, neviđeni Re) — R² (više = bolje)\n{'='*70}")
    print(res[["tag", "R2_all", "R2_U_x", "R2_U_y", "R2_p"]].to_string(index=False))
    print(f"\nSačuvano: {out.relative_to(ROOT)}")
    print("Zaključak: ako collocation R²_all > data_only R²_all, fizika pomaže u sparse režimu.")


if __name__ == "__main__":
    main()
