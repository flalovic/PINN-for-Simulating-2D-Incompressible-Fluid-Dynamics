# Physics-Informed Neural Networks for the 2D Lid-Driven Cavity

A study of **Physics-Informed Neural Networks (PINNs)** for the two-dimensional
incompressible **lid-driven cavity** flow. Models are trained on
**OpenFOAM-generated** reference data while simultaneously being constrained by
the incompressible **Navier–Stokes equations** through a physics-informed loss.

The central question of the project is **how much the physics loss helps when the
network is pushed away from the labeled data** — either *spatially* (a region of
the domain is hidden during training) or in *parameter space* (Reynolds numbers
beyond the training range). Across every experiment the answer is consistent:
**the harder the generalization, the more the physics term pays off.**

<p align="center">
  <img src="assets/anim_re_1100.gif" width="750">
</p>

---

## Table of Contents

1. [Overview](#overview)
2. [The Physics: Lid-Driven Cavity](#the-physics-lid-driven-cavity)
3. [Method](#method)
   - [PINN Formulation](#pinn-formulation)
   - [Architectures](#architectures)
4. [Data Generation Pipeline (OpenFOAM)](#data-generation-pipeline-openfoam)
5. [Data Sampling](#data-sampling)
6. [Experiments & Results](#experiments--results)
   - [1. Held-out Reynolds numbers (in-range)](#1-held-out-reynolds-numbers-in-range)
   - [2. Held-out spatial region](#2-held-out-spatial-region)
   - [3. Reynolds-number extrapolation](#3-reynolds-number-extrapolation)
   - [Key finding](#key-finding)
7. [Repository Structure](#repository-structure)
8. [Reproducing the Results](#reproducing-the-results)
9. [Technologies](#technologies)

---

## Overview

The model is a coordinate network that maps flow parameters to flow variables:

**Inputs** &nbsp;→&nbsp; `(t, Re, x, y)` &nbsp;—&nbsp; time, Reynolds number, spatial coordinates

**Outputs** &nbsp;→&nbsp; `(Uₓ, U_y, p)` &nbsp;—&nbsp; horizontal velocity, vertical velocity, pressure

Training combines two objectives:

- a **supervised data loss** against OpenFOAM ground-truth snapshots, and
- a **physics-informed loss** that penalizes the residual of the incompressible
  Navier–Stokes equations, evaluated at randomly sampled *collocation* points
  (including regions where no labeled data exists).

A scalar weight `c_physics` balances the two. Sweeping this weight is the backbone
of every experiment below.

---

## The Physics: Lid-Driven Cavity

The lid-driven cavity is the classic CFD benchmark: a square box filled with fluid,
where three walls are stationary (`noSlip`) and the **top wall (the "lid") slides
horizontally** at a fixed velocity. The moving lid drags the fluid and sets up a
primary recirculating vortex, with secondary corner vortices that strengthen as
inertia grows.

The behavior is governed by the **Reynolds number**

```
Re = U · L / ν
```

where `U` is the lid velocity, `L` the cavity side length, and `ν` the kinematic
viscosity. In this project `U = 1`, `L = 1`, so a target `Re` is obtained purely
by setting `ν = 1 / Re`.

The transient, incompressible Navier–Stokes equations solved (in non-dimensional
form) are:

```
Continuity:   ∂Uₓ/∂x + ∂U_y/∂y = 0
Momentum-x:   ∂Uₓ/∂t + Uₓ ∂Uₓ/∂x + U_y ∂Uₓ/∂y = −∂p/∂x + (1/Re)(∂²Uₓ/∂x² + ∂²Uₓ/∂y²)
Momentum-y:   ∂U_y/∂t + Uₓ ∂U_y/∂x + U_y ∂U_y/∂y = −∂p/∂y + (1/Re)(∂²U_y/∂x² + ∂²U_y/∂y²)
```

These three residuals form the physics loss.

---

## Method

### PINN Formulation

The loss (`src/loss.py`, `NavierStokesLoss`) is:

```
L = L_data  +  c_physics · L_physics
```

- **`L_data`** — mean-squared error between predicted and OpenFOAM `(Uₓ, U_y, p)`.
- **`L_physics`** — mean of the squared continuity + x/y-momentum residuals.

Key implementation details:

- All derivatives (`∂/∂t`, `∂/∂x`, `∂/∂y`, and second-order `∂²/∂x²`, `∂²/∂y²`)
  are computed by **automatic differentiation** (`torch.autograd.grad` with
  `create_graph=True`), not finite differences.
- The network is trained on **z-score normalized** inputs/outputs. The physics
  loss therefore **de-normalizes** predictions and **rescales gradients by the
  per-variable standard deviations** before assembling the residuals, so the PDE
  is enforced in physical units.
- **Collocation training** (`src/train.py`, `train_collocation`): the data loss is
  evaluated on labeled points, while the physics residual is evaluated on
  `n_collocation` points sampled uniformly across the domain each step. This
  constrains the solution even where there is no labeled data — which is exactly
  what makes physics help in the held-out and extrapolation experiments.
- The best checkpoint is selected on **validation data loss** (`best_model.pth`),
  and full checkpoints (model/optimizer/scheduler/history) are written every epoch
  for resumability.

### Architectures

Two model families are implemented in `src/models.py`:

**1. MLP PINN** — the standard coordinate network: 4 hidden layers of width 256 with
`Tanh` activations (smoothness is required so the second derivatives in the physics
loss are well-defined). This is the model used in the reported experiments.

**2. GINO (Geometry-Informed Neural Operator)** — a full neural-operator pipeline
implemented from scratch:

```
MeshGNN  →  GNO encoder  →  FNO kernel  →  GNO decoder  →  MLP head
```

- **`MeshGNN`** — EdgeConv-style message passing on a k-NN graph over the mesh
  nodes.
- **`GNOEncoder` / `GNODecoder`** — kernel-integral (graph-neural-operator) layers
  that transfer features between the scattered mesh points and a uniform latent
  grid; the decoder keeps the query coordinates differentiable so the physics loss
  still applies.
- **`FNO2d` / `SpectralConv2d`** — a Fourier Neural Operator core with
  complex-valued spectral convolutions (truncated to `fno_modes` Fourier modes) on
  the latent grid.

The GINO is configured through `config/architecture/gino.yaml`
(`GINO.from_config`).

---

## Data Generation Pipeline (OpenFOAM)

Ground-truth data is produced by an automated pipeline
(`openfoam_setup/generate_dataset.py`) that treats OpenFOAM as a black-box CFD
engine and sweeps it across Reynolds numbers.

For each target `Re`, the pipeline:

1. **Fills OpenFOAM case templates** (`openfoam_setup/templates/*.template`) from
   YAML configs (`config/sampling/`), substituting geometry, mesh resolution, lid
   velocity, viscosity `ν = U·L/Re`, and time controls:
   - `U.template` → `0/U` (lid boundary condition)
   - `physicalProperties.template` → `constant/physicalProperties` (`ν`)
   - `blockMeshDict.template` → `system/blockMeshDict` (domain + mesh)
   - `controlDict.template` → `system/controlDict` (`deltaT`, `endTime`, write interval)
2. **Builds the mesh** with `blockMesh`.
3. **Runs the transient solver** `icoFoam` (incompressible, laminar, Newtonian;
   PISO pressure–velocity coupling).
4. **Extracts 2D fields with PyVista**: opens the case, slices the thin 3D slab back
   to a 2D plane at `z ≈ 0`, reads `U` and `p` at cell centers for every written
   timestep, and flattens to a DataFrame `(time, re, x, y, U_x, U_y, p)`.
5. **Splits by Reynolds number** into train / validation / test using a
   **stratified** split (`pd.qcut` bins over the Re range), so each split spans the
   full range rather than clustering. The result is written to `data/` as
   `*_train.csv`, `*_valid.csv`, `*_test.csv`.

Sampling settings (`config/sampling/`):

| Config | Purpose | Re range | Mesh |
| :--- | :--- | :--- | :--- |
| `reynolds.yaml` | Re sampling + split (50 samples, 70/15/15) | 100–1000 | — |
| `train.yaml` | train/valid case geometry | — | 64 × 64 |
| `test.yaml` | test case geometry (finer) | — | 128 × 128 |
| `animation.yaml` | dense-in-time case for animations | — | 128 × 128 |

Run it with:

```bash
cd openfoam_setup
python generate_dataset.py -tr train.yaml -te test.yaml -re reynolds.yaml -o data
```

> Requires a working OpenFOAM installation (developed against **OpenFOAM v2412**)
> with `blockMesh`, `icoFoam`, and `foamCleanTutorials` on the path.

---

## Data Sampling

The full OpenFOAM train/validation sets are very large (100+ MB CSVs). Because the
PINN advantage is clearest in the **sparse-data regime**, the raw data is
sub-sampled into fractions (`data/sampled_data/frac_1`, `frac_5`, `frac_10` ≈ 1 %,
5 %, 10 %). Notebook `01_data_analysis.ipynb` inspects these splits. All reported
experiments train on the smallest, hardest **`frac_1`** set (~2.4 k training rows).

---

## Experiments & Results

Every experiment sweeps the physics weight
`c_physics ∈ {0, 0.01, 0.05, 0.1, 0.5, 1.0}` (with `c_physics = 0` being a pure
data-driven baseline) under identical seeds, and compares generalization.

### 1. Held-out Reynolds numbers (in-range)

The base test: evaluate the best PINN on Reynolds numbers held out of training but
**within** the sampled range (100–1000).

| Variable     |          MAE |         RMSE |           R² |  Relative L2 |
| :----------- | -----------: | -----------: | -----------: | -----------: |
| **Uₓ**       |     0.002370 |     0.005236 |     0.999086 |     0.030240 |
| **U_y**      |     0.002089 |     0.004637 |     0.998711 |     0.035896 |
| **Pressure** |     0.000830 |     0.005112 |     0.985978 |     0.113105 |
| **Overall**  | **0.001763** | **0.005001** | **0.998457** | **0.039254** |

In-range interpolation is essentially solved (overall **R² = 0.9985**).

*Notebooks: `02` (training/ablation), `03` (field visualization), `04` (metric plots).*

### 2. Held-out spatial region

A rectangular region in the **center of the cavity is removed from the training
data**, and the model is scored on its ability to reconstruct that hidden region.
Three window sizes are tested (25 %, 49 %, 64 % of the domain area). This isolates
the physics loss's ability to fill *spatial* gaps.

**R² inside the held-out region vs. `c_physics`:**

| Held-out area | c=0 (data only) |  c=0.1 | best (c) |
| :------------ | --------------: | -----: | :------- |
| 25 %          |           0.954 |  0.978 | **0.983** (c=0.5) |
| 49 %          |           0.865 |  0.948 | **0.959** (c=0.5) |
| 64 %          |           0.457 |  0.455 | **0.603** (c=1.0) |

The larger the hidden region, the more the physics term matters: at the 64 % window
the pure-data model collapses to **R² = 0.46**, while physics lifts it to **0.60**.

*Notebooks: `02` (training/ablation), `03` (field + animation), `04` (metric plots).
Runs in `heldout_runs/`, metrics in `results/heldout_results_{25,49,64}.csv`.*

### 3. Reynolds-number extrapolation

Models are trained on Re ∈ [100, 1000] and evaluated on Re ∈ {1200, 1400, 1600,
1800, 2000} — **entirely outside the training range**.

**Overall R² vs. `c_physics`** (whole extrapolation test set):

| c_physics | 0 | 0.01 | 0.05 | **0.1** | 0.5 | 1.0 |
| :-------- | ----: | ---: | ---: | ----: | ---: | ---: |
| R²_all    | 0.804 | 0.818 | 0.820 | **0.836** | 0.820 | 0.831 |

**Degradation with distance from the training range** (R², best physics model
`c=0.1` vs. data-only `c=0`):

| Re   | 1200 | 1400 | 1600 | 1800 | 2000 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| physics (c=0.1) | 0.951 | 0.906 | 0.842 | 0.764 | 0.677 |
| data only (c=0) | 0.946 | 0.894 | 0.814 | 0.715 | 0.605 |
| **gain**        | +0.005 | +0.012 | +0.028 | +0.049 | **+0.072** |

The physics benefit **grows monotonically the further the model extrapolates** —
from +0.005 just past the boundary to +0.072 at Re = 2000.

*Notebooks: `05` (training/ablation), `06` (metric plots), `07` (field + 4×3
animation). Runs in `extrapolation_runs/`, metrics in
`results/re_extrapolation_results.csv`.*

### Key finding

Across all three settings the story is the same: **when labeled data is plentiful
and the query is easy, the physics loss barely matters; as the task moves away from
the data — a larger spatial hole, or a Reynolds number farther outside the training
range — the physics-informed loss provides a growing, measurable improvement in
generalization.** This is the core empirical result of the project.

---

## Repository Structure

```
.
├── openfoam_setup/                                # OpenFOAM data-generation pipeline
│   ├── generate_dataset.py                        # templating → blockMesh → icoFoam → PyVista → CSV
│   ├── templates/                                 # parameterized OpenFOAM dictionaries
│   └── cavity/                                    # OpenFOAM case (written into by the pipeline)
├── config/
│   ├── sampling/                                  # Re range + case geometry (train/test/animation)
│   └── architecture/gino.yaml                     # GINO hyperparameters
├── src/
│   ├── models.py                                  # PINN (MLP) and GINO (MeshGNN/GNO/FNO)
│   ├── loss.py                                    # NavierStokesLoss (data + PDE residual)
│   ├── train.py                                   # standard and collocation training loops
│   ├── dataloader.py                              # data loading, per-state datasets
│   ├── utils.py                                   # metrics, normalization, collocation sampling, boxes
│   └── visuals.py                                 # field plots + animations
├── 01_data_analysis.ipynb                         # dataset inspection & sampling
├── 02_heldout_training_evaluation.ipynb           # spatial held-out: train + c_physics ablation
├── 03_heldout_test_visualisation.ipynb            # spatial held-out: field + animation
├── 04_heldout_metrics_plots.ipynb                 # spatial held-out: metric plots
├── 05_re_extrapolation_training_evaluation.ipynb  # Re extrapolation: train + ablation
├── 06_re_extrapolation_metrics_plots.ipynb        # Re extrapolation: metric plots
├── 07_re_extrapolation_visualisation.ipynb        # Re extrapolation: field + 4×3 animation
├── data/                                          # raw + sampled datasets
├── heldout_runs/  extrapolation_runs/             # trained checkpoints + configs + metrics
└── results/                                       # aggregated metrics + animations
```

---

## Reproducing the Results

1. **(Optional) Generate the dataset** — requires OpenFOAM:
   ```bash
   cd openfoam_setup
   python generate_dataset.py -tr train.yaml -te test.yaml -re reynolds.yaml -o data
   ```
2. **Analyze / sample the data** — `01_data_analysis.ipynb`.
3. **Held-out spatial-region experiment** — run `02` → `03` → `04`.
4. **Reynolds-extrapolation experiment** — run `05` → `06` → `07`.

Each training notebook sweeps `c_physics`, writes per-run checkpoints and metrics to
`*_runs/`, and aggregates results into `results/`. The visualization notebooks load
the best physics model and the `c=0` baseline and render field comparisons and
animations.

---

## Technologies

- **Python**, **PyTorch** — models, autograd-based physics loss, training
- **OpenFOAM** (`blockMesh`, `icoFoam`) — ground-truth CFD data
- **PyVista** — mesh reading and 2D slice extraction
- **NumPy / pandas** — data handling
- **Matplotlib / seaborn** — plots, metric heatmaps, animations
- **scikit-learn** — stratified Reynolds-number splitting
