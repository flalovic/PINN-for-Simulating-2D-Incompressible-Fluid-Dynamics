import os
import torch
import tempfile


import numpy as np
import pandas as pd
import yaml

import src.utils as utils
import matplotlib.patches as patches

from PIL import Image
from matplotlib import pyplot as plt


def _compute_vorticity(U_x, U_y, x_unique, y_unique):
    """
    Izračunava polje vorticiteta (vrtložnosti) na osnovu komponenti brzine.
    """
    dUy_dx = np.gradient(U_y, x_unique, axis=1)
    dUx_dy = np.gradient(U_x, y_unique, axis=0)
    return dUy_dx - dUx_dy

def animate_error(model, df_orig, re_value, mean, std, device, fps=2, output_file=None):
    """
    Generiše i animira prostornu raspodjelu greške modela kroz vrijeme za zadati Reynolds-ov broj.
    """
    data_re = df_orig[df_orig['re'] == re_value]
    time_steps = sorted(data_re['time'].unique())

    if len(time_steps) < 2:
        print(f"Nema dovoljno vremenskih koraka za Re={re_value}")
        return

    data_first = data_re[data_re['time'] == time_steps[0]]
    x_unique = np.array(sorted(data_first['x'].unique()))
    y_unique = np.array(sorted(data_first['y'].unique()))
    nx, ny = len(x_unique), len(y_unique)

    all_errors = []
    for t in time_steps:
        data = data_re[data_re['time'] == t]

        input_data = data[['time', 're', 'x', 'y']].values
        target_data = data[['U_x', 'U_y', 'p']].values

        input_norm = (input_data - mean[['time', 're', 'x', 'y']].values) / std[['time', 're', 'x', 'y']].values
        target_norm = (target_data - mean[['U_x', 'U_y', 'p']].values) / std[['U_x', 'U_y', 'p']].values

        input_tensor = torch.tensor(input_norm, dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            pred_norm = model(input_tensor).cpu().numpy()

        pred = pred_norm * std[['U_x', 'U_y', 'p']].values + mean[['U_x', 'U_y', 'p']].values
        target = target_norm * std[['U_x', 'U_y', 'p']].values + mean[['U_x', 'U_y', 'p']].values

        X = data['x'].values.reshape(ny, nx)
        Y = data['y'].values.reshape(ny, nx)

        error_u_x = np.abs(target[:, 0] - pred[:, 0]).reshape(ny, nx)
        error_u_y = np.abs(target[:, 1] - pred[:, 1]).reshape(ny, nx)
        error_p   = np.abs(target[:, 2] - pred[:, 2]).reshape(ny, nx)

        all_errors.append({
            'X': X, 'Y': Y,
            'error_u_x': error_u_x, 'error_u_y': error_u_y, 'error_p': error_p,
            'time': t
        })

    err_ux_all = [e['error_u_x'] for e in all_errors]
    err_uy_all = [e['error_u_y'] for e in all_errors]
    err_p_all  = [e['error_p']   for e in all_errors]

    vmin_ux, vmax_ux = min(e.min() for e in err_ux_all), max(e.max() for e in err_ux_all)
    vmin_uy, vmax_uy = min(e.min() for e in err_uy_all), max(e.max() for e in err_uy_all)
    vmin_p,  vmax_p  = min(e.min() for e in err_p_all),  max(e.max() for e in err_p_all)

    with tempfile.TemporaryDirectory() as tmp_dir:
        frame_files = []

        print(f"Generiši {len(all_errors)} frame-ova greške...")
        for idx, frame in enumerate(all_errors):
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            X, Y = frame['X'], frame['Y']

            cf0 = axes[0, 0].contourf(X, Y, frame['error_u_x'], levels=15, cmap='hot', vmin=vmin_ux, vmax=vmax_ux)
            axes[0, 0].set_title('Greška U_x', pad=10)
            axes[0, 0].set_ylabel('y')
            axes[0, 0].set_aspect('equal')
            axes[0, 0].set_xlim(x_unique.min(), x_unique.max())
            axes[0, 0].set_ylim(y_unique.min(), y_unique.max())
            plt.colorbar(cf0, ax=axes[0, 0])

            cf1 = axes[0, 1].contourf(X, Y, frame['error_u_y'], levels=15, cmap='hot', vmin=vmin_uy, vmax=vmax_uy)
            axes[0, 1].set_title('Greška U_y', pad=10)
            axes[0, 1].set_aspect('equal')
            axes[0, 1].set_xlim(x_unique.min(), x_unique.max())
            axes[0, 1].set_ylim(y_unique.min(), y_unique.max())
            plt.colorbar(cf1, ax=axes[0, 1])

            cf2 = axes[1, 0].contourf(X, Y, frame['error_p'], levels=15, cmap='hot', vmin=vmin_p, vmax=vmax_p)
            axes[1, 0].set_title('Greška p', pad=10)
            axes[1, 0].set_xlabel('x')
            axes[1, 0].set_ylabel('y')
            axes[1, 0].set_aspect('equal')
            axes[1, 0].set_xlim(x_unique.min(), x_unique.max())
            axes[1, 0].set_ylim(y_unique.min(), y_unique.max())
            plt.colorbar(cf2, ax=axes[1, 0])

            axes[1, 1].axis('off')
            stats_text = f"""
                Vrijeme: {frame['time']:.3f}s
                Reynolds: {re_value}

                MAE U_x: {frame['error_u_x'].mean():.6f}
                MAE U_y: {frame['error_u_y'].mean():.6f}
                MAE p:   {frame['error_p'].mean():.6f}

                Frejm: {idx+1}/{len(all_errors)}
            """
            axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                            verticalalignment='center')

            fig.suptitle(f'Animacija greške - Re={re_value}', fontsize=14, y=0.97)
            plt.tight_layout(pad=1.5, rect=[0, 0, 1, 0.95])

            frame_path = os.path.join(tmp_dir, f'frame_error_{idx:04d}.png')
            fig.savefig(frame_path, dpi=100, bbox_inches='tight')
            frame_files.append(frame_path)
            plt.close(fig)

            if (idx + 1) % max(1, len(all_errors) // 5) == 0:
                print(f"  ✓ {idx+1}/{len(all_errors)} frame-ova generisano")

        if output_file:
            print(f"\nKombinujem slike u GIF ({output_file})...")
            images = [Image.open(f) for f in frame_files]
            duration = int(1000 / fps)
            images[0].save(
                output_file,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=0,
                optimize=False
            )
            print(f"✓ GIF greške spreman: {output_file}")
            return output_file

def animate_flow(df, re_value, output_file="flow_animation.gif", fps=3):
    """
    Kreira animaciju polja brzina, pritiska i vorticiteta kroz vrijeme te ih čuva kao GIF.
    """
    data_re = df[df['re'] == re_value]
    time_steps = sorted(data_re['time'].unique())

    if len(time_steps) < 2:
        print(f"Nema dovoljno vremenskih koraka za Re={re_value}")
        return

    data_first = data_re[data_re['time'] == time_steps[0]]
    x_unique = np.array(sorted(data_first['x'].unique()))
    y_unique = np.array(sorted(data_first['y'].unique()))
    nx, ny = len(x_unique), len(y_unique)

    all_frames = []
    for t in time_steps:
        data = data_re[data_re['time'] == t]

        X   = data['x'].values.reshape(ny, nx)
        Y   = data['y'].values.reshape(ny, nx)
        U_x = data['U_x'].values.reshape(ny, nx)
        U_y = data['U_y'].values.reshape(ny, nx)
        P   = data['p'].values.reshape(ny, nx)
        vorticity = _compute_vorticity(U_x, U_y, x_unique, y_unique)

        all_frames.append({
            'X': X, 'Y': Y, 'U_x': U_x, 'U_y': U_y, 'P': P,
            'vorticity': vorticity, 'time': t
        })

    speed_all = [np.sqrt(f['U_x']**2 + f['U_y']**2) for f in all_frames]
    p_all     = [f['P'] for f in all_frames]
    vort_all  = [f['vorticity'] for f in all_frames]

    vmin_speed, vmax_speed = min(s.min() for s in speed_all), max(s.max() for s in speed_all)
    vmin_p,     vmax_p     = min(p.min() for p in p_all),     max(p.max() for p in p_all)
    vort_abs = max(np.abs(v).max() for v in vort_all)

    with tempfile.TemporaryDirectory() as tmp_dir:
        frame_files = []

        print(f"Generiši {len(all_frames)} frame-ova...")
        for idx, frame in enumerate(all_frames):
            fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))

            X, Y   = frame['X'], frame['Y']
            U_x, U_y = frame['U_x'], frame['U_y']
            P      = frame['P']
            speed  = np.sqrt(U_x**2 + U_y**2)
            vorticity = frame['vorticity']

            cf0 = axes[0].contourf(X, Y, speed, levels=20, cmap='viridis', vmin=vmin_speed, vmax=vmax_speed)
            axes[0].streamplot(x_unique, y_unique, U_x, U_y, color='white', linewidth=0.8, density=1.5, arrowsize=1.0)
            axes[0].set_xlabel('x')
            axes[0].set_ylabel('y')
            axes[0].set_title(f'Polje brzina - t = {frame["time"]:.3f}s', pad=10)
            axes[0].set_aspect('equal')
            axes[0].set_xlim(x_unique.min(), x_unique.max())
            axes[0].set_ylim(y_unique.min(), y_unique.max())
            plt.colorbar(cf0, ax=axes[0], label='Brzina [m/s]')

            cf1 = axes[1].contourf(X, Y, P, levels=20, cmap='RdBu_r', vmin=vmin_p, vmax=vmax_p)
            axes[1].set_xlabel('x')
            axes[1].set_ylabel('y')
            axes[1].set_title(f'Pritisak - t = {frame["time"]:.3f}s', pad=10)
            axes[1].set_aspect('equal')
            axes[1].set_xlim(x_unique.min(), x_unique.max())
            axes[1].set_ylim(y_unique.min(), y_unique.max())
            plt.colorbar(cf1, ax=axes[1], label='Pritisak [Pa]')

            cf2 = axes[2].contourf(X, Y, vorticity, levels=20, cmap='RdBu_r', vmin=-vort_abs, vmax=vort_abs)
            axes[2].set_xlabel('x')
            axes[2].set_ylabel('y')
            axes[2].set_title(f'Vorticitet - t = {frame["time"]:.3f}s', pad=10)
            axes[2].set_aspect('equal')
            axes[2].set_xlim(x_unique.min(), x_unique.max())
            axes[2].set_ylim(y_unique.min(), y_unique.max())
            plt.colorbar(cf2, ax=axes[2], label='Vorticitet [1/s]')

            fig.suptitle(f'Animacija toka - Re={re_value} (frejm {idx+1}/{len(all_frames)})', fontsize=14, y=0.96)
            plt.tight_layout(pad=1.2, rect=[0, 0, 1, 0.93])

            frame_path = os.path.join(tmp_dir, f'frame_{idx:04d}.png')
            fig.savefig(frame_path, dpi=100, bbox_inches='tight')
            frame_files.append(frame_path)
            plt.close(fig)

            if (idx + 1) % max(1, len(all_frames) // 5) == 0:
                print(f"  ✓ {idx+1}/{len(all_frames)} frame-ova generisano")

        images = [Image.open(f) for f in frame_files]
        duration = int(1000 / fps)
        images[0].save(output_file, save_all=True, append_images=images[1:], duration=duration, loop=0, optimize=False)
        print(f"✓ GIF spreman: {output_file}")

    return output_file

def plot_velocity_and_pressure(df, time_step, re_value, title_prefix=""):
    """
    Vizuelizuje trenutno stanje polja brzine sa strujnicama, pritiska i vorticiteta za fiksirani trenutak i Re.
    """
    data = df[(df['time'] == time_step) & (df['re'] == re_value)].copy()

    if len(data) == 0:
        print(f"Nema podataka za time={time_step}, Re={re_value}")
        return

    x_unique = np.array(sorted(data['x'].unique()))
    y_unique = np.array(sorted(data['y'].unique()))
    nx, ny = len(x_unique), len(y_unique)

    X   = data['x'].values.reshape(ny, nx)
    Y   = data['y'].values.reshape(ny, nx)
    U_x = data['U_x'].values.reshape(ny, nx)
    U_y = data['U_y'].values.reshape(ny, nx)
    P   = data['p'].values.reshape(ny, nx)
    vorticity = _compute_vorticity(U_x, U_y, x_unique, y_unique)

    fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))

    speed = np.sqrt(U_x**2 + U_y**2)
    cf = axes[0].contourf(X, Y, speed, levels=20, cmap='viridis')
    axes[0].streamplot(x_unique, y_unique, U_x, U_y, color='white', linewidth=0.8, density=1.5, arrowsize=1.0)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title(f'{title_prefix} Polje brzina - Re={re_value}, t={time_step}', pad=10)
    axes[0].set_aspect('equal')
    axes[0].set_xlim(x_unique.min(), x_unique.max())
    axes[0].set_ylim(y_unique.min(), y_unique.max())
    plt.colorbar(cf, ax=axes[0], label='Brzina [m/s]')

    cf = axes[1].contourf(X, Y, P, levels=20, cmap='RdBu_r')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title(f'{title_prefix} Pritisak - Re={re_value}, t={time_step}', pad=10)
    axes[1].set_aspect('equal')
    axes[1].set_xlim(x_unique.min(), x_unique.max())
    axes[1].set_ylim(y_unique.min(), y_unique.max())
    plt.colorbar(cf, ax=axes[1], label='Pritisak [Pa]')

    vort_abs = np.abs(vorticity).max() if np.abs(vorticity).max() > 0 else 1.0
    cf = axes[2].contourf(X, Y, vorticity, levels=20, cmap='RdBu_r', vmin=-vort_abs, vmax=vort_abs)
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_title(f'{title_prefix} Vorticitet - Re={re_value}, t={time_step}', pad=10)
    axes[2].set_aspect('equal')
    axes[2].set_xlim(x_unique.min(), x_unique.max())
    axes[2].set_ylim(y_unique.min(), y_unique.max())
    plt.colorbar(cf, ax=axes[2], label='Vorticitet [1/s]')

    plt.tight_layout(pad=1.2)
    plt.show()

def plot_evolution_in_time(df, re_value):
    """
    Kreira mrežu podgrafika prikazujući vremensku evoluciju polja brzina, pritiska i vorticiteta.
    """
    num_plots = len(df['time'].unique())
    data_re = df[df['re'] == re_value]
    time_steps = sorted(data_re['time'].unique())

    step_size = max(1, len(time_steps) // num_plots)
    selected_times = time_steps[::step_size][:num_plots]

    fig, axes = plt.subplots(num_plots, 3, figsize=(18, 3.8 * num_plots))
    if num_plots == 1:
        axes = [axes]

    for idx, t in enumerate(selected_times):
        data = data_re[data_re['time'] == t]

        x_unique = np.array(sorted(data['x'].unique()))
        y_unique = np.array(sorted(data['y'].unique()))
        nx, ny = len(x_unique), len(y_unique)

        X   = data['x'].values.reshape(ny, nx)
        Y   = data['y'].values.reshape(ny, nx)
        U_x = data['U_x'].values.reshape(ny, nx)
        U_y = data['U_y'].values.reshape(ny, nx)
        P   = data['p'].values.reshape(ny, nx)
        vorticity = _compute_vorticity(U_x, U_y, x_unique, y_unique)

        speed = np.sqrt(U_x**2 + U_y**2)
        cf = axes[idx][0].contourf(X, Y, speed, levels=15, cmap='viridis')
        axes[idx][0].streamplot(x_unique, y_unique, U_x, U_y, color='white', linewidth=0.8, density=1.2, arrowsize=0.9)
        axes[idx][0].set_title(f't = {t:.3f}s, Re={re_value}', pad=10)
        axes[idx][0].set_ylabel('y')
        axes[idx][0].set_aspect('equal')
        axes[idx][0].set_xlim(x_unique.min(), x_unique.max())
        axes[idx][0].set_ylim(y_unique.min(), y_unique.max())
        plt.colorbar(cf, ax=axes[idx][0])

        cf = axes[idx][1].contourf(X, Y, P, levels=15, cmap='RdBu_r')
        axes[idx][1].set_title(f'Pritisak - t = {t:.3f}s', pad=10)
        axes[idx][1].set_aspect('equal')
        axes[idx][1].set_xlim(x_unique.min(), x_unique.max())
        axes[idx][1].set_ylim(y_unique.min(), y_unique.max())
        plt.colorbar(cf, ax=axes[idx][1])

        vort_abs = np.abs(vorticity).max()
        cf = axes[idx][2].contourf(X, Y, vorticity, levels=15, cmap='RdBu_r', vmin=-vort_abs, vmax=vort_abs)
        axes[idx][2].set_title(f'Vorticitet - t = {t:.3f}s', pad=10)
        axes[idx][2].set_aspect('equal')
        axes[idx][2].set_xlim(x_unique.min(), x_unique.max())
        axes[idx][2].set_ylim(y_unique.min(), y_unique.max())
        plt.colorbar(cf, ax=axes[idx][2])

    fig.suptitle(f'Evolucija toka u vremenu - Re={re_value}', fontsize=14, y=0.99)
    plt.tight_layout(pad=1.5, rect=[0, 0, 1, 0.97])
    plt.show()

def compare_predictions(model, df, time_step, re_value, mean, std, device):
    """
    Izračunava, ispisuje metriku (MAE) i vizuelizuje apsolutne i relativne greške predikcije modela.
    """
    data = df[(df['time'] == time_step) & (df['re'] == re_value)].copy()

    if len(data) == 0:
        print(f"Nema podataka za time={time_step}, Re={re_value}")
        return

    input_data  = data[['time', 're', 'x', 'y']].values
    target_data = data[['U_x', 'U_y', 'p']].values

    input_norm  = (input_data  - mean[['time', 're', 'x', 'y']].values) / std[['time', 're', 'x', 'y']].values
    target_norm = (target_data - mean[['U_x', 'U_y', 'p']].values)      / std[['U_x', 'U_y', 'p']].values

    input_tensor  = torch.tensor(input_norm,  dtype=torch.float32).to(device)
    target_tensor = torch.tensor(target_norm, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(input_tensor).cpu().numpy()
    target = target_tensor.cpu().numpy()

    pred_denorm   = pred   * std[['U_x', 'U_y', 'p']].values + mean[['U_x', 'U_y', 'p']].values
    target_denorm = target * std[['U_x', 'U_y', 'p']].values + mean[['U_x', 'U_y', 'p']].values

    x_unique = np.array(sorted(data['x'].unique()))
    y_unique = np.array(sorted(data['y'].unique()))
    nx, ny = len(x_unique), len(y_unique)

    X = data['x'].values.reshape(ny, nx)
    Y = data['y'].values.reshape(ny, nx)

    U_x_true = target_denorm[:, 0].reshape(ny, nx)
    U_y_true = target_denorm[:, 1].reshape(ny, nx)
    P_true   = target_denorm[:, 2].reshape(ny, nx)

    U_x_pred = pred_denorm[:, 0].reshape(ny, nx)
    U_y_pred = pred_denorm[:, 1].reshape(ny, nx)
    P_pred   = pred_denorm[:, 2].reshape(ny, nx)

    error_u_x = np.abs(U_x_true - U_x_pred)
    error_u_y = np.abs(U_y_true - U_y_pred)
    error_p   = np.abs(P_true   - P_pred)

    print(f"MAE U_x: {error_u_x.mean():.6f}")
    print(f"MAE U_y: {error_u_y.mean():.6f}")
    print(f"MAE p:   {error_p.mean():.6f}")

    fig, axes = plt.subplots(3, 2, figsize=(12, 11))

    for idx, (error, label) in enumerate([(error_u_x, 'U_x'), (error_u_y, 'U_y'), (error_p, 'p')]):
        cf = axes[idx, 0].contourf(X, Y, error, levels=15, cmap='hot')
        axes[idx, 0].set_title(f'Greška {label}', pad=10)
        axes[idx, 0].set_ylabel('y')
        axes[idx, 0].set_aspect('equal')
        axes[idx, 0].set_xlim(x_unique.min(), x_unique.max())
        axes[idx, 0].set_ylim(y_unique.min(), y_unique.max())
        plt.colorbar(cf, ax=axes[idx, 0])

        if label == 'p':
            rel_error = 100 * error / (np.abs(P_true) + 1e-8)
        else:
            true_val = U_x_true if label == 'U_x' else U_y_true
            rel_error = 100 * error / (np.abs(true_val) + 1e-8)

        cf = axes[idx, 1].contourf(X, Y, rel_error, levels=15, cmap='hot')
        axes[idx, 1].set_title(f'Relativna greška {label} (%)', pad=10)
        axes[idx, 1].set_aspect('equal')
        axes[idx, 1].set_xlim(x_unique.min(), x_unique.max())
        axes[idx, 1].set_ylim(y_unique.min(), y_unique.max())
        plt.colorbar(cf, ax=axes[idx, 1])

    plt.suptitle(f'Greška predviđanja - Re={re_value}, t={time_step}', fontsize=14, y=0.98)
    plt.tight_layout(pad=1.5, rect=[0, 0, 1, 0.96])
    plt.show()

def animate_truth_vs_pred(model, df_orig, re_value, mean, std, device, fps=2, output_file="truth_vs_pred.gif"):
    """
    Generiše i animira direktno uporedno poređenje (Ground Truth vs Predikcija)
    kroz sve dostupne vremenske trenutke za zadati Reynolds-ov broj i čuva rezultat kao GIF.
    """
    data_re = df_orig[df_orig['re'] == re_value]
    time_steps = sorted(data_re['time'].unique())

    if len(time_steps) < 1:
        print(f"Nema podataka za Re={re_value}")
        return

    data_first = data_re[data_re['time'] == time_steps[0]]
    x_unique = np.array(sorted(data_first['x'].unique()))
    y_unique = np.array(sorted(data_first['y'].unique()))
    nx, ny = len(x_unique), len(y_unique)

    all_frames = []
    print(f"Izračunavanje predikcija za {len(time_steps)} vremenskih koraka...")

    for t in time_steps:
        data = data_re[data_re['time'] == t]

        input_data  = data[['time', 're', 'x', 'y']].values
        target_data = data[['U_x', 'U_y', 'p']].values

        input_norm = (input_data - mean[['time', 're', 'x', 'y']].values) / std[['time', 're', 'x', 'y']].values

        input_tensor = torch.tensor(input_norm, dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            pred_norm = model(input_tensor).cpu().numpy()

        pred_denorm = pred_norm * std[['U_x', 'U_y', 'p']].values + mean[['U_x', 'U_y', 'p']].values

        X = data['x'].values.reshape(ny, nx)
        Y = data['y'].values.reshape(ny, nx)

        U_x_true   = target_data[:, 0].reshape(ny, nx)
        U_y_true   = target_data[:, 1].reshape(ny, nx)
        P_true     = target_data[:, 2].reshape(ny, nx)
        speed_true = np.sqrt(U_x_true**2 + U_y_true**2)
        vort_true  = _compute_vorticity(U_x_true, U_y_true, x_unique, y_unique)

        U_x_pred   = pred_denorm[:, 0].reshape(ny, nx)
        U_y_pred   = pred_denorm[:, 1].reshape(ny, nx)
        P_pred     = pred_denorm[:, 2].reshape(ny, nx)
        speed_pred = np.sqrt(U_x_pred**2 + U_y_pred**2)
        vort_pred  = _compute_vorticity(U_x_pred, U_y_pred, x_unique, y_unique)

        all_frames.append({
            'X': X, 'Y': Y, 'time': t,
            'speed_true': speed_true, 'speed_pred': speed_pred,
            'U_x_true': U_x_true, 'U_x_pred': U_x_pred,
            'U_y_true': U_y_true, 'U_y_pred': U_y_pred,
            'P_true': P_true, 'P_pred': P_pred,
            'vort_true': vort_true, 'vort_pred': vort_pred
        })

    max_vabs_ux   = max(max(np.abs(f['U_x_true']).max(), np.abs(f['U_x_pred']).max()) for f in all_frames)
    max_vabs_uy   = max(max(np.abs(f['U_y_true']).max(), np.abs(f['U_y_pred']).max()) for f in all_frames)
    max_vabs_p    = max(max(np.abs(f['P_true']).max(),   np.abs(f['P_pred']).max())   for f in all_frames)
    max_vabs_vort = max(max(np.abs(f['vort_true']).max(),  np.abs(f['vort_pred']).max())  for f in all_frames)
    
    max_speed = max(max(f['speed_true'].max(), f['speed_pred'].max()) for f in all_frames)
    min_speed = min(min(f['speed_true'].min(), f['speed_pred'].min()) for f in all_frames)

    with tempfile.TemporaryDirectory() as tmp_dir:
        frame_files = []

        print(f"Renderovanje frejmova...")
        for idx, frame in enumerate(all_frames):
            
            fields = [
                (frame['speed_true'], frame['speed_pred'], 'Speed [m/s]',     'viridis', min_speed, max_speed),
                (frame['U_x_true'],   frame['U_x_pred'],   'U_x [m/s]',       'RdBu_r',  -max_vabs_ux, max_vabs_ux),
                (frame['U_y_true'],   frame['U_y_pred'],   'U_y [m/s]',       'RdBu_r',  -max_vabs_uy, max_vabs_uy),
                (frame['P_true'],     frame['P_pred'],     'Pritisak [Pa]',    'RdBu_r',  -max_vabs_p,  max_vabs_p),
                (frame['vort_true'],  frame['vort_pred'],  'Vorticitet [1/s]', 'RdBu_r',  -max_vabs_vort, max_vabs_vort),
            ]

            n_rows = len(fields)
            fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4.2 * n_rows))

            X, Y = frame['X'], frame['Y']

            for row, (truth, pred, label, cmap, vmin, vmax) in enumerate(fields):
                cf_t = axes[row, 0].contourf(X, Y, truth, levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
                axes[row, 0].set_title(f'Ground Truth — {label}', pad=10)
                axes[row, 0].set_ylabel('y')
                axes[row, 0].set_aspect('equal')
                axes[row, 0].set_xlim(x_unique.min(), x_unique.max())
                axes[row, 0].set_ylim(y_unique.min(), y_unique.max())
                plt.colorbar(cf_t, ax=axes[row, 0], label=label)

                cf_p = axes[row, 1].contourf(X, Y, pred, levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
                axes[row, 1].set_title(f'Predikcija — {label}', pad=10)
                axes[row, 1].set_aspect('equal')
                axes[row, 1].set_xlim(x_unique.min(), x_unique.max())
                axes[row, 1].set_ylim(y_unique.min(), y_unique.max())
                plt.colorbar(cf_p, ax=axes[row, 1], label=label)

                if row == 0:
                    axes[row, 0].streamplot(x_unique, y_unique, frame['U_x_true'], frame['U_y_true'],
                                            color='white', linewidth=0.8, density=1.2, arrowsize=0.9)
                    axes[row, 1].streamplot(x_unique, y_unique, frame['U_x_pred'], frame['U_y_pred'],
                                            color='white', linewidth=0.8, density=1.2, arrowsize=0.9)

            axes[-1, 0].set_xlabel('x')
            axes[-1, 1].set_xlabel('x')

            fig.suptitle(f'Ground Truth vs Predikcija — Re={re_value}, t={frame["time"]:.3f}s', fontsize=14, y=0.99)
            plt.tight_layout(pad=1.5, rect=[0, 0, 1, 0.98])

            frame_path = os.path.join(tmp_dir, f'frame_tvsp_{idx:04d}.png')
            fig.savefig(frame_path, dpi=100, bbox_inches='tight')
            frame_files.append(frame_path)
            plt.close(fig)

            if (idx + 1) % max(1, len(all_frames) // 5) == 0:
                print(f"  ✓ {idx+1}/{len(all_frames)} frejmova generisano")

        images = [Image.open(f) for f in frame_files]
        duration = int(1000 / fps)
        images[0].save(output_file, save_all=True, append_images=images[1:], duration=duration, loop=0, optimize=False)
        print(f"✓ GIF uspješno kreiran: {output_file}")

    return output_file

def animate_flow_tmp(ANIM_OUT, BOX, test_df, RE_VALUE, model_phys, model_nophys, best_phys,
                     mean, std, device, config_path="config/sampling/animation.yaml"):
    """
    Animira rekonstrukciju polja kroz vrijeme za tri izvora u mrezi 4x3:
    kolone = Ground truth / PINN (fizika) / mreza bez fizike,
    redovi = Brzina (+ strujnice) / U_x / U_y / Pritisak.

    Prati isti obrazac kao `animate_flow` i `animate_error`: iterira PRAVE vremenske
    korake (bez interpolacije), renderuje svaki frejm u PNG i slaze ih u GIF preko PIL-a.
    fps se cita iz `config_path` (config/sampling/animation.yaml) kao
    1 / (delta_t * write_interval), a granice osa iz `domain` sekcije.
    """
    input_cols = ['time', 're', 'x', 'y']
    target_cols = ['U_x', 'U_y', 'p']

    with open(config_path) as f:
        anim_cfg = yaml.safe_load(f)

    domain = anim_cfg['domain']
    time_control = anim_cfg['system']['time_control']

    # fps za reprodukciju u realnom vremenu: 1 frame svakih (delta_t * write_interval) sekundi
    fps = 1.0 / (time_control['delta_t'] * time_control['write_interval'])
    x_min, x_max = domain['x_min'], domain['x_max']
    y_min, y_max = domain['y_min'], domain['y_max']

    data_re = test_df[test_df['re'] == RE_VALUE]
    time_steps = sorted(data_re['time'].unique())

    if len(time_steps) < 2:
        print(f"Nema dovoljno vremenskih koraka za Re={RE_VALUE}")
        return

    data_first = data_re[data_re['time'] == time_steps[0]]
    x_unique = np.array(sorted(data_first['x'].unique()))
    y_unique = np.array(sorted(data_first['y'].unique()))
    nx, ny = len(x_unique), len(y_unique)

    def predict(model, data):
        input_norm = (data[input_cols].values - mean[input_cols].values) / std[input_cols].values
        input_tensor = torch.tensor(input_norm, dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            pred_norm = model(input_tensor).cpu().numpy()
        return pred_norm * std[target_cols].values + mean[target_cols].values

    sources = [
        ("Ground truth", None),
        (f"PINN (fizika, c={best_phys})", model_phys),
        ("Bez fizike (c=0)", model_nophys),
    ]

    # Predracunaj polja (U_x, U_y, p) po vremenu za sva tri izvora
    all_frames = []
    for t in time_steps:
        data = data_re[data_re['time'] == t]
        X = data['x'].values.reshape(ny, nx)
        Y = data['y'].values.reshape(ny, nx)
        true = data[target_cols].values

        per_source = []
        for _, model in sources:
            vals = true if model is None else predict(model, data)
            per_source.append((
                vals[:, 0].reshape(ny, nx),
                vals[:, 1].reshape(ny, nx),
                vals[:, 2].reshape(ny, nx),
            ))
        all_frames.append({'X': X, 'Y': Y, 'time': t, 'fields': per_source})

    # Zajednicke skale po komponenti (konzistentne boje kroz panele i frejmove).
    # Brzina: viridis [0, max]; U_x/U_y/p: RdBu_r simetricno oko nule.
    def comp_absmax(idx):
        return max(np.abs(f['fields'][s][idx]).max()
                   for f in all_frames for s in range(len(sources)))

    speed_max = max(np.sqrt(f['fields'][s][0] ** 2 + f['fields'][s][1] ** 2).max()
                    for f in all_frames for s in range(len(sources)))
    a_ux, a_uy, a_p = comp_absmax(0), comp_absmax(1), comp_absmax(2)

    components = [
        ("speed", "Brzina [m/s]", "viridis", 0.0, float(speed_max)),
        ("U_x", "U_x [m/s]", "RdBu_r", -float(a_ux), float(a_ux)),
        ("U_y", "U_y [m/s]", "RdBu_r", -float(a_uy), float(a_uy)),
        ("p", "Pritisak [Pa]", "RdBu_r", -float(a_p), float(a_p)),
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        frame_files = []

        print(f"Generiši {len(all_frames)} frame-ova (fps={fps:.2f} iz {config_path})...")
        for idx, frame in enumerate(all_frames):
            X, Y = frame['X'], frame['Y']
            fig, axes = plt.subplots(len(components), len(sources), figsize=(18, 22),
                                     constrained_layout=True)

            for row, (key, label, cmap, vmin, vmax) in enumerate(components):
                levels = np.linspace(vmin, vmax, 21)
                cf = None

                for col, (title, _) in enumerate(sources):
                    ax = axes[row, col]
                    U_x, U_y, P = frame['fields'][col]

                    if key == "speed":
                        field = np.sqrt(U_x ** 2 + U_y ** 2)
                        cf = ax.contourf(X, Y, field, levels=levels, cmap=cmap, extend="max")
                        ax.streamplot(x_unique, y_unique, U_x, U_y,
                                      color="white", linewidth=0.7, density=1.2, arrowsize=0.9)
                    else:
                        field = {"U_x": U_x, "U_y": U_y, "p": P}[key]
                        cf = ax.contourf(X, Y, field, levels=levels, cmap=cmap, extend="both")

                    ax.add_patch(patches.Rectangle(
                        (BOX["x_min"], BOX["y_min"]),
                        BOX["x_max"] - BOX["x_min"], BOX["y_max"] - BOX["y_min"],
                        fill=False, edgecolor="red", linewidth=1.5, linestyle="--",
                    ))
                    ax.set_aspect("equal")
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)

                    if row == 0:
                        ax.set_title(title)
                    if row == len(components) - 1:
                        ax.set_xlabel("x")
                    if col == 0:
                        ax.set_ylabel(label)

                fig.colorbar(cf, ax=axes[row, :].tolist(), label=label, fraction=0.02, pad=0.02)

            fig.suptitle(f"Re={RE_VALUE:.1f}, t={frame['time']:.2f}s", fontsize=15)

            frame_path = os.path.join(tmp_dir, f"frame_flow_{idx:04d}.png")
            fig.savefig(frame_path, dpi=90, bbox_inches="tight")
            frame_files.append(frame_path)
            plt.close(fig)

            if (idx + 1) % max(1, len(all_frames) // 5) == 0:
                print(f"  ✓ {idx+1}/{len(all_frames)} frame-ova generisano")

        images = [Image.open(f) for f in frame_files]
        duration = int(1000 / fps)
        images[0].save(ANIM_OUT, save_all=True, append_images=images[1:],
                       duration=duration, loop=0, optimize=False)
        print(f"✓ GIF spreman: {ANIM_OUT}")

    return ANIM_OUT

def evaluate(model, test_df_orig, mean, std, device, output_dir="eval_animations", fps=2):
    """
    Izvršava evaluaciju modela kroz generisanje animacija grešaka za sve jedinstvene Reynolds-ove brojeve u test setu.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    re_values = sorted(test_df_orig['re'].unique())

    for re_val in re_values:
        output_file_path = os.path.join(output_dir, f"error_animation_Re_{int(re_val)}.gif")
        try:
            animate_error(model, test_df_orig, re_val, mean, std, device, fps, output_file_path)
        except Exception as e:
            print(f"Greška tokom evaluacije za Re={re_val}: {e}")