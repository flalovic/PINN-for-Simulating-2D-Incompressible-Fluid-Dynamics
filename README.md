# Physics-Informed Neural Network for the Lid-Driven Cavity Problem

A Physics-Informed Neural Network (PINN) for solving the two-dimensional incompressible lid-driven cavity flow. The model learns the transient solution of the Navier–Stokes equations using OpenFOAM-generated data while simultaneously enforcing the governing physical laws through a physics-informed loss.

---

## Flow Animation

<p align="center">
  <img src="assets/anim_re_1100.gif" width="750">
</p>

---

## Overview

The model predicts the flow variables from the following inputs:

* Reynolds number
* Time
* Spatial coordinates $(x,y)$

The predicted quantities are:

* Horizontal velocity ($U_x$)
* Vertical velocity ($U_y$)
* Pressure ($p$)

The training objective combines a supervised data loss with a physics-informed loss based on the incompressible Navier–Stokes equations.

---

## Test Performance

The model was evaluated on Reynolds numbers not used during training.

| Variable     |          MAE |         RMSE |           R² |  Relative L2 |
| :----------- | -----------: | -----------: | -----------: | -----------: |
| **Uₓ**       |     0.002370 |     0.005236 |     0.999086 |     0.030240 |
| **Uᵧ**       |     0.002089 |     0.004637 |     0.998711 |     0.035896 |
| **Pressure** |     0.000830 |     0.005112 |     0.985978 |     0.113105 |
| **Overall**  | **0.001763** | **0.005001** | **0.998457** | **0.039254** |

---

## Features

* Physics-Informed Neural Network (PINN)
* OpenFOAM-based dataset generation
* Prediction of transient velocity and pressure fields
* Combined supervised and physics-informed training
* Automatic dataset generation pipeline
* Visualization and animation utilities

---

## Technologies

* Python
* PyTorch
* OpenFOAM
* PyVista
* NumPy
* Pandas
* Matplotlib

---

## Results

The proposed PINN accurately reconstructs the transient velocity and pressure fields over a range of Reynolds numbers. On the unseen test set, the model achieves an overall **R² score of 0.99846**, demonstrating strong agreement with the reference OpenFOAM simulations and excellent generalization performance.
