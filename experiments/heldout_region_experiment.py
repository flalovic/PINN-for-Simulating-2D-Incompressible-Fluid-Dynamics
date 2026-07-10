"""
Held-out region eksperiment: presudni test da li fizika pomaže tamo gdje NEMA podataka.

Iz treninga se UKLANJAJU svi labelirani uzorci unutar prostornog boksa (x,y) ∈ box,
i to za SVE snapshot-ove (sve Re i vrijeme). Kolokacione tačke se i dalje uzorkuju po
CIJELOM domenu (uključujući boks), pa fizika ograničava rješenje i u praznini.
Greška se mjeri ODVOJENO unutar i izvan boksa, za data-only vs collocation-PINN.

Očekivanje: ako fizika pomaže, R² UNUTAR boksa raste kod collocation modela u odnosu
na data-only, dok R² IZVAN boksa ostaje sličan.

Pokretanje (iz korijena repo-a, na GPU mašini):
    python -m experiments.heldout_region_experiment --epochs 300 --weights 0 0.1 \
        --box 0.3 0.7 0.3 0.7
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
    set_seed(42)  # identičan start -> fer poređenje
    model = PINN(len(INPUT_COLS), len(TARGET_COLS)).to(device)
    criterion = NavierStokesLoss(c_physics, mean, std)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)

    train_collocation(
        model, train_dl, valid_dl, criterion, optimizer, scheduler,
        device, epochs, run_dir, INPUT_COLS, ranges, mean, std, n_collocation=n_coll,
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", default="frac_1")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-collocation", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=32768)
    ap.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.1])
    ap.add_argument("--box", type=float, nargs=4, default=[0.3, 0.7, 0.3, 0.7],
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    args = ap.parse_args()

    box = dict(zip(["x_min", "x_max", "y_min", "y_max"], args.box))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | frac: {args.frac} | epochs: {args.epochs} | box: {box}")

    # --- sparse trening/validacija, pa IZBACI boks iz oba ---
    train_df, valid_df, _ = load_data(ROOT / "data" / "real_data" / args.frac, "data")
    _, train_out = utils.split_by_box(train_df, box)      # zadrži samo IZVAN boksa
    _, valid_out = utils.split_by_box(valid_df, box)
    print(f"Trening: {len(train_df)} -> {len(train_out)} (uklonjeno {len(train_df)-len(train_out)} iz boksa)")

    # normalizacija iz maskiranog treninga (to model zapravo vidi)
    mean, std = train_out.mean(), train_out.std()
    tr_n = utils.normalize_data(train_out, mean, std)
    va_n = utils.normalize_data(valid_out, mean, std)
    train_dl, valid_dl, _ = gen_dataloaders(tr_n, va_n, va_n, INPUT_COLS, TARGET_COLS, args.batch_size)

    # kolokacija po CIJELOM domenu (x,y ∈ [0,1]) -> pokriva i prazan boks
    ranges = utils.get_domain_ranges(train_df, INPUT_COLS, overrides={"x": (0.0, 1.0), "y": (0.0, 1.0)})

    # --- PUNI test skup, podijeljen na unutar/izvan boksa ---
    _, _, test_df = load_data(ROOT / "data" / "raw_data", "full_data")
    test_in, test_out = utils.split_by_box(test_df, box)
    print(f"Test: unutar boksa={len(test_in)}  izvan boksa={len(test_out)}")

    results = []
    for w in args.weights:
        tag = "data_only" if w == 0 else f"coll_cphys_{w}"
        print(f"\n{'='*70}\n{tag} (c_physics={w})\n{'='*70}")
        run_dir = utils.create_run_directory(label=f"heldout_{args.frac}_{tag}")
        model = run_one(w, train_dl, valid_dl, ranges, mean, std, device,
                        args.epochs, args.n_collocation, run_dir)

        for region, sub in [("inside_box", test_in), ("outside_box", test_out)]:
            m = utils.evaluate_model(model, sub, INPUT_COLS, TARGET_COLS, mean, std, device)
            row = {"c_physics": w, "tag": tag, "region": region, "n_points": len(sub)}
            row.update(utils.flatten_metrics(m))   # sve metrike, sve komponente
            results.append(row)

    res = pd.DataFrame(results)
    out = ROOT / "experiments" / f"heldout_results_{args.frac}.csv"
    res.to_csv(out, index=False)

    print(f"\n{'='*70}\nR² po regionu (INSIDE = prazan boks, presudan)\n{'='*70}")
    print(res[["tag", "region", "R2_all", "R2_U_x", "R2_U_y", "R2_p"]].to_string(index=False))

    # jasan sud: poređenje unutar boksa
    inside = res[res.region == "inside_box"].set_index("tag")["R2_all"]
    if "data_only" in inside.index:
        base = inside["data_only"]
        print(f"\nUNUTAR boksa  R²_all(data_only) = {base:.4f}")
        for tag, val in inside.items():
            if tag != "data_only":
                d = val - base
                verdict = "fizika POMAŽE" if d > 0.005 else ("nema razlike" if d > -0.005 else "fizika ŠKODI")
                print(f"  {tag}: {val:.4f}  (Δ={d:+.4f})  -> {verdict}")
    print(f"\nSačuvano: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
