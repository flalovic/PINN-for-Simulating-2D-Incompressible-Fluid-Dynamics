import os
import torch
import tempfile

import numpy as np
import pandas as pd

from PIL import Image
from matplotlib import pyplot as plt

# ovdje se nalaze funkcije za vizuelizaciju i animaciju:
# animate_error()
# animate_flow()
# plot_velocity_and_pressure()
# plot_evolution_in_time()
# plot_compare_predictions()
# evaluate()



# Animacija greske kroz vrijeme
def animate_error(model, df_orig, re_value, mean, std, device, fps=2, output_file=None):
    """
    Animira grešku modela kroz vrijeme
    """
    data_re = df_orig[df_orig['re'] == re_value]
    time_steps = sorted(data_re['time'].unique())
    
    if len(time_steps) < 2:
        print(f"Nema dovoljno vremenskih koraka za Re={re_value}")
        return
    
    # Kreiraj grid za prvi vremenski korak
    data_first = data_re[data_re['time'] == time_steps[0]]
    x_unique = sorted(data_first['x'].unique())
    y_unique = sorted(data_first['y'].unique())
    nx, ny = len(x_unique), len(y_unique)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Pripremi sve podatke unaprijed
    all_errors = []
    for t in time_steps:
        data = data_re[data_re['time'] == t]
        
        # Pripremi ulaze
        input_data = data[['time', 're', 'x', 'y']].values
        target_data = data[['U_x', 'U_y', 'p']].values
        
        # Normalizuj
        input_norm = (input_data - mean[['time', 're', 'x', 'y']].values) / std[['time', 're', 'x', 'y']].values
        target_norm = (target_data - mean[['U_x', 'U_y', 'p']].values) / std[['U_x', 'U_y', 'p']].values
        
        # Predviđanje
        input_tensor = torch.tensor(input_norm, dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            pred_norm = model(input_tensor).cpu().numpy()
        
        # Denormalizuj
        pred = pred_norm * std[['U_x', 'U_y', 'p']].values + mean[['U_x', 'U_y', 'p']].values
        target = target_norm * std[['U_x', 'U_y', 'p']].values + mean[['U_x', 'U_y', 'p']].values
        
        # Greške
        X = data['x'].values.reshape(ny, nx)
        Y = data['y'].values.reshape(ny, nx)
        
        error_u_x = np.abs(target[:, 0] - pred[:, 0]).reshape(ny, nx)
        error_u_y = np.abs(target[:, 1] - pred[:, 1]).reshape(ny, nx)
        error_p = np.abs(target[:, 2] - pred[:, 2]).reshape(ny, nx)
        
        all_errors.append({
            'X': X, 'Y': Y, 
            'error_u_x': error_u_x, 'error_u_y': error_u_y, 'error_p': error_p,
            'time': t
        })
    
    # Minmax
    err_ux_all = [e['error_u_x'] for e in all_errors]
    err_uy_all = [e['error_u_y'] for e in all_errors]
    err_p_all = [e['error_p'] for e in all_errors]
    
    vmin_ux, vmax_ux = min([e.min() for e in err_ux_all]), max([e.max() for e in err_ux_all])
    vmin_uy, vmax_uy = min([e.min() for e in err_uy_all]), max([e.max() for e in err_uy_all])
    vmin_p, vmax_p = min([e.min() for e in err_p_all]), max([e.max() for e in err_p_all])
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        frame_files = []
        
        # Generiši sve slike
        print(f"Generiši {len(all_errors)} frame-ova greške...")
        for idx, frame in enumerate(all_errors):
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            X, Y = frame['X'], frame['Y']
            
            # U_x
            cf0 = axes[0, 0].contourf(X, Y, frame['error_u_x'], levels=15, cmap='hot', vmin=vmin_ux, vmax=vmax_ux)
            axes[0, 0].set_title(f'Greška U_x')
            axes[0, 0].set_ylabel('y')
            plt.colorbar(cf0, ax=axes[0, 0])
            
            # U_y
            cf1 = axes[0, 1].contourf(X, Y, frame['error_u_y'], levels=15, cmap='hot', vmin=vmin_uy, vmax=vmax_uy)
            axes[0, 1].set_title(f'Greška U_y')
            plt.colorbar(cf1, ax=axes[0, 1])
            
            # p
            cf2 = axes[1, 0].contourf(X, Y, frame['error_p'], levels=15, cmap='hot', vmin=vmin_p, vmax=vmax_p)
            axes[1, 0].set_title(f'Greška p')
            axes[1, 0].set_xlabel('x')
            axes[1, 0].set_ylabel('y')
            plt.colorbar(cf2, ax=axes[1, 0])
            
            # Statistika
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
            
            fig.suptitle(f'Animacija greške - Re={re_value}', fontsize=14)
            plt.tight_layout()
            
            # Spremi kao sliku
            frame_path = os.path.join(tmp_dir, f'frame_error_{idx:04d}.png')
            fig.savefig(frame_path, dpi=100, bbox_inches='tight')
            frame_files.append(frame_path)
            
            plt.close(fig)
            
            # Progress
            if (idx + 1) % max(1, len(all_errors) // 5) == 0:
                print(f"  ✓ {idx+1}/{len(all_errors)} frame-ova generisano")
        
        # Kombinuj slike u GIF ako je output_file specificiran
        if output_file:
            print(f"\nKombinujem slike u GIF ({output_file})...")
            images = [Image.open(f) for f in frame_files]
            
            duration = int(1000 / fps)  # Vrijeme po frame-u u ms
            images[0].save(
                output_file,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=0,
                optimize=False
            )
            
            print(f"✓ GIF greške spreman: {output_file}")
            print(f"  • Frejmova: {len(all_errors)}")
            print(f"  • Brzina: {fps} fps")
            print(f"  • Trajanje: {len(all_errors) / fps:.1f}s")
            
            return output_file
        else:
            print(f"✓ Slike generisane - {len(all_errors)} frame-ova")

# Animacija toka kroz vrijeme - generise slike i slaze u GIF
def animate_flow(df, re_value, output_file="flow_animation.gif", fps=3):
    """
    Kreira animaciju brzinskog polja i pritiska kroz vrijeme:
    1. Generiše sve frame-ove kao slike
    2. Kombinuje ih u GIF
    
    Args:
        df: DataFrame sa podacima
        re_value: Reynolds broj
        output_file: Putanja GIF fajla (npr. 'animation.gif')
        fps: Broj frejmova po sekundi
    """
    data_re = df[df['re'] == re_value]
    time_steps = sorted(data_re['time'].unique())
    
    if len(time_steps) < 2:
        print(f"Nema dovoljno vremenskih koraka za Re={re_value}")
        return
    
    # Kreiraj grid za prvi vremenski korak
    data_first = data_re[data_re['time'] == time_steps[0]]
    x_unique = sorted(data_first['x'].unique())
    y_unique = sorted(data_first['y'].unique())
    nx, ny = len(x_unique), len(y_unique)
    
    # Pripremi sve podatke unaprijed
    all_frames = []
    for t in time_steps:
        data = data_re[data_re['time'] == t]
        
        X = data['x'].values.reshape(ny, nx)
        Y = data['y'].values.reshape(ny, nx)
        U_x = data['U_x'].values.reshape(ny, nx)
        U_y = data['U_y'].values.reshape(ny, nx)
        P = data['p'].values.reshape(ny, nx)
        
        all_frames.append({
            'X': X, 'Y': Y, 'U_x': U_x, 'U_y': U_y, 'P': P, 'time': t
        })
    
    # Minmax za normalizaciju boja
    speed_all = [np.sqrt(f['U_x']**2 + f['U_y']**2) for f in all_frames]
    p_all = [f['P'] for f in all_frames]
    
    vmin_speed, vmax_speed = min([s.min() for s in speed_all]), max([s.max() for s in speed_all])
    vmin_p, vmax_p = min([p.min() for p in p_all]), max([p.max() for p in p_all])
    
    # Kreiraj privremeni direktorijum za slike
    with tempfile.TemporaryDirectory() as tmp_dir:
        frame_files = []
        
        # Generiši sve slike
        print(f"Generiši {len(all_frames)} frame-ova...")
        for idx, frame in enumerate(all_frames):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            X, Y = frame['X'], frame['Y']
            U_x, U_y = frame['U_x'], frame['U_y']
            P = frame['P']
            speed = np.sqrt(U_x**2 + U_y**2)
            
            # Brzina (lijevo)
            cf0 = axes[0].contourf(X, Y, speed, levels=20, cmap='viridis', vmin=vmin_speed, vmax=vmax_speed)
            axes[0].quiver(X[::2, ::2], Y[::2, ::2], U_x[::2, ::2], U_y[::2, ::2], alpha=0.6)
            axes[0].set_xlabel('x')
            axes[0].set_ylabel('y')
            axes[0].set_title(f'Brzinsko polje - t = {frame["time"]:.3f}s')
            plt.colorbar(cf0, ax=axes[0], label='Brzina [m/s]')
            
            # Pritisak (desno)
            cf1 = axes[1].contourf(X, Y, P, levels=20, cmap='RdBu_r', vmin=vmin_p, vmax=vmax_p)
            axes[1].set_xlabel('x')
            axes[1].set_ylabel('y')
            axes[1].set_title(f'Pritisak - t = {frame["time"]:.3f}s')
            plt.colorbar(cf1, ax=axes[1], label='Pritisak [Pa]')
            
            fig.suptitle(f'Animacija toka - Re={re_value} (frejm {idx+1}/{len(all_frames)})', fontsize=14)
            plt.tight_layout()
            
            # Spremi kao sliku
            frame_path = os.path.join(tmp_dir, f'frame_{idx:04d}.png')
            fig.savefig(frame_path, dpi=100, bbox_inches='tight')
            frame_files.append(frame_path)
            
            plt.close(fig)
            
            # Progress
            if (idx + 1) % max(1, len(all_frames) // 5) == 0:
                print(f"  ✓ {idx+1}/{len(all_frames)} frame-ova generirano")
        
        # Kombinuj slike u GIF
        print(f"\nKombinujem slike u GIF ({output_file})...")
        images = [Image.open(f) for f in frame_files]
        
        duration = int(1000 / fps)  # Vrijeme po frame-u u ms
        images[0].save(
            output_file,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,
            optimize=False
        )
        
        print(f"✓ GIF spreman: {output_file}")
        print(f"  • Frejmova: {len(all_frames)}")
        print(f"  • Brzina: {fps} fps")
        print(f"  • Trajanje: {len(all_frames) / fps:.1f}s")

    return output_file


# Vizuelizacija za odredjeni vremenski trenutak i Reynolds-ov broj
def plot_velocity_and_pressure(df, time_step, re_value, title_prefix=""):
    """Vizuelizuje brzinsko polje i pritisak"""
    data = df[(df['time'] == time_step) & (df['re'] == re_value)].copy()
    
    if len(data) == 0:
        print(f"Nema podataka za time={time_step}, Re={re_value}")
        return
    
    # Kreiraj grid
    x_unique = sorted(data['x'].unique())
    y_unique = sorted(data['y'].unique())
    
    nx, ny = len(x_unique), len(y_unique)
    
    X = data['x'].values.reshape(ny, nx)
    Y = data['y'].values.reshape(ny, nx)
    U_x = data['U_x'].values.reshape(ny, nx)
    U_y = data['U_y'].values.reshape(ny, nx)
    P = data['p'].values.reshape(ny, nx)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Brzinsko polje
    ax = axes[0]
    speed = np.sqrt(U_x**2 + U_y**2)
    cf = ax.contourf(X, Y, speed, levels=20, cmap='viridis')
    ax.quiver(X[::2, ::2], Y[::2, ::2], U_x[::2, ::2], U_y[::2, ::2], alpha=0.6)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{title_prefix} Brzinsko polje - Re={re_value}, t={time_step}')
    plt.colorbar(cf, ax=ax, label='Brzina [m/s]')
    
    # Pritisak
    ax = axes[1]
    cf = ax.contourf(X, Y, P, levels=20, cmap='RdBu_r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{title_prefix} Pritisak - Re={re_value}, t={time_step}')
    plt.colorbar(cf, ax=ax, label='Pritisak [Pa]')
    
    plt.tight_layout()
    plt.show()



def plot_evolution_in_time(df, re_value):
    """Prikaži kako se brzina i pritisak mijenjaju tijekom vremena"""
    num_plots = len(df['time'].unique())
    data_re = df[df['re'] == re_value]
    time_steps = sorted(data_re['time'].unique())
    
    # Odaberi ravnomjerno raspoređene vremenske korake
    step_size = max(1, len(time_steps) // num_plots)
    selected_times = time_steps[::step_size][:num_plots]
    
    fig, axes = plt.subplots(num_plots, 2, figsize=(12, 3*num_plots))
    if num_plots == 1:
        axes = [axes]
    
    for idx, t in enumerate(selected_times):
        data = data_re[data_re['time'] == t]
        
        x_unique = sorted(data['x'].unique())
        y_unique = sorted(data['y'].unique())
        nx, ny = len(x_unique), len(y_unique)
        
        X = data['x'].values.reshape(ny, nx)
        Y = data['y'].values.reshape(ny, nx)
        U_x = data['U_x'].values.reshape(ny, nx)
        U_y = data['U_y'].values.reshape(ny, nx)
        P = data['p'].values.reshape(ny, nx)
        
        # Brzina
        speed = np.sqrt(U_x**2 + U_y**2)
        cf = axes[idx][0].contourf(X, Y, speed, levels=15, cmap='viridis')
        axes[idx][0].set_title(f't = {t:.3f}s, Re={re_value}')
        axes[idx][0].set_ylabel('y')
        plt.colorbar(cf, ax=axes[idx][0])
        
        # Pritisak
        cf = axes[idx][1].contourf(X, Y, P, levels=15, cmap='RdBu_r')
        axes[idx][1].set_title(f'Pritisak - t = {t:.3f}s')
        plt.colorbar(cf, ax=axes[idx][1])
    
    fig.suptitle(f'Evolucija toka u vremenu - Re={re_value}', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.show()


def compare_predictions(model, df, time_step, re_value, mean, std, device):
    """Vizuelizuje razliku između predviđanja modela i stvarnih podataka"""
    data = df[(df['time'] == time_step) & (df['re'] == re_value)].copy()
    
    if len(data) == 0:
        print(f"Nema podataka za time={time_step}, Re={re_value}")
        return
    
    # Pripremi ulazne podatke (nenormalizovane)
    input_data = data[['time', 're', 'x', 'y']].values
    target_data = data[['U_x', 'U_y', 'p']].values
    
    # Normalizuj kao što je rađeno u treningu
    input_norm = (input_data - mean[['time', 're', 'x', 'y']].values) / std[['time', 're', 'x', 'y']].values
    target_norm = (target_data - mean[['U_x', 'U_y', 'p']].values) / std[['U_x', 'U_y', 'p']].values
    
    # Pretvori u tensore
    input_tensor = torch.tensor(input_norm, dtype=torch.float32).to(device)
    target_tensor = torch.tensor(target_norm, dtype=torch.float32).to(device)
    
    # Predviđanje
    model.eval()
    with torch.no_grad():
        pred = model(input_tensor).cpu().numpy()
    target = target_tensor.cpu().numpy()
    
    # Denormalizuj za prikaz
    pred_denorm = pred * std[['U_x', 'U_y', 'p']].values + mean[['U_x', 'U_y', 'p']].values
    target_denorm = target * std[['U_x', 'U_y', 'p']].values + mean[['U_x', 'U_y', 'p']].values
    
    # Kreiraj grid
    x_unique = sorted(data['x'].unique())
    y_unique = sorted(data['y'].unique())
    nx, ny = len(x_unique), len(y_unique)
    
    X = data['x'].values.reshape(ny, nx)
    Y = data['y'].values.reshape(ny, nx)
    
    U_x_true = target_denorm[:, 0].reshape(ny, nx)
    U_y_true = target_denorm[:, 1].reshape(ny, nx)
    P_true = target_denorm[:, 2].reshape(ny, nx)
    
    U_x_pred = pred_denorm[:, 0].reshape(ny, nx)
    U_y_pred = pred_denorm[:, 1].reshape(ny, nx)
    P_pred = pred_denorm[:, 2].reshape(ny, nx)
    
    # Greške
    error_u_x = np.abs(U_x_true - U_x_pred)
    error_u_y = np.abs(U_y_true - U_y_pred)
    error_p = np.abs(P_true - P_pred)
    
    print(f"MAE U_x: {error_u_x.mean():.6f}")
    print(f"MAE U_y: {error_u_y.mean():.6f}")
    print(f"MAE p: {error_p.mean():.6f}")
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    # Apsolutne greške
    for idx, (error, label) in enumerate([(error_u_x, 'U_x'), (error_u_y, 'U_y'), (error_p, 'p')]):
        cf = axes[idx, 0].contourf(X, Y, error, levels=15, cmap='hot')
        axes[idx, 0].set_title(f'Greška {label}')
        axes[idx, 0].set_ylabel('y')
        plt.colorbar(cf, ax=axes[idx, 0])
        
        # Relativna greška (%)
        if label == 'p':
            rel_error = 100 * error / (np.abs(P_true) + 1e-8)
        else:
            true_val = U_x_true if label == 'U_x' else U_y_true
            rel_error = 100 * error / (np.abs(true_val) + 1e-8)
        
        cf = axes[idx, 1].contourf(X, Y, rel_error, levels=15, cmap='hot')
        axes[idx, 1].set_title(f'Relativna greška {label} (%)')
        plt.colorbar(cf, ax=axes[idx, 1])
    
    plt.suptitle(f'Greška predviđanja - Re={re_value}, t={time_step}', fontsize=14)
    plt.tight_layout()
    plt.show()

# evaluacija na proslijedjenom skupu
def evaluate(model, test_df_orig, mean, std, device, output_dir="eval_animations", fps=2):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    re_values = sorted(test_df_orig['re'].unique())
    
    for re_val in re_values:
        output_file_path = os.path.join(output_dir, f"error_animation_Re_{int(re_val)}.gif")
        try:
            animate_error(model, test_df_orig, re_val, mean, std, device, fps, output_file_path)
        except Exception as e:
            print(f"Greška za Re={re_val}: {e}")