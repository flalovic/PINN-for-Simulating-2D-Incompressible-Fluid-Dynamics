import torch
from torch.utils.data import TensorDataset, DataLoader

import os
import numpy as np
import pandas as pd

from pathlib import Path
from utils import check_regions

def sample_by_region(df: pd.DataFrame, regions: list[dict[str, float]]) -> pd.DataFrame:
    """
    Funkcija koja po regionima uzorkuje tacke! Regioni moraju biti disjunktni!    
    Primjer region mape:
        lijevi_kvadrat = {
            'x_min': 0.0,
            'x_max': 0.5,
            'y_min': 0.0,
            'y_max': 0.5,
            'dropout': 0.5,  # 50% tacaka ce biti
            'random_state': 42,  # za reproducibilnost
            'groupby_cols': ['re', 'time']  # opcionalno, default je ['re']
        }

    Primjer liste regiona:
        e = 0.003125  # 2 cell layers
        regions = [
            {   # Lid — only non-zero BC, most important
                'x_min': 0.0,     'x_max': 1,
                'y_min': 1 - e, 'y_max': 1,
                'dropout': 0.0,
                'random_state': 42,
                'groupby_cols': ['re']
            },
            {   # Bottom wall — no-slip
                'x_min': 0.0, 'x_max': 1,
                'y_min': 0.0, 'y_max': e,
                'dropout': 0.1,
                'random_state': 42,
                'groupby_cols': ['re']
            },
            {   # Left wall — no-slip, excludes corners already covered above/below
                'x_min': 0.0, 'x_max': e,
                'y_min': e,   'y_max': 1 - e,
                'dropout': 0.1,
                'random_state': 42,
                'groupby_cols': ['re']
            },
            {   # Right wall — no-slip
                'x_min': 1 - e, 'x_max': 1,
                'y_min': e,        'y_max': 1 - e,
                'dropout': 0.1,
                'random_state': 42,
                'groupby_cols': ['re']
            },
            {   # Interior — vortex core, high redundancy
                'x_min': e,       'x_max': 1 - e,
                'y_min': e,       'y_max': 1 - e,
                'dropout': 0.7,
                'random_state': 42,
                'groupby_cols': ['re']
            },
        ]
    """
    assert check_regions(regions) is True, "Regions must be disjoint!"
    sampled_df = pd.DataFrame(columns=df.columns)
    for region in regions:
        x_min, x_max = region['x_min'], region['x_max']
        y_min, y_max = region['y_min'], region['y_max']
        dropout = region['dropout']
        random_state = region['random_state']
        groupby_cols = region.get('groupby_cols', ['re'])
        
        bounded_df = df[(df['x'] >= x_min) & (df['x'] <= x_max) & (df['y'] >= y_min) & (df['y'] <= y_max)]
        
        grouped_df = bounded_df.groupby(groupby_cols, group_keys=False)

        new_sampled_df = grouped_df.sample(frac=1-dropout, random_state=random_state)
        sampled_df = pd.concat([sampled_df, new_sampled_df])

    return sampled_df


def load_data(file_path: Path, file_prefix: str):
    """
    Učitavamo trening, validacioni i test skup iz CSV fajlova.
    """
    train_df = pd.read_csv(file_path / f"{file_prefix}_train.csv")
    valid_df = pd.read_csv(file_path / f"{file_prefix}_valid.csv")
    test_df = pd.read_csv(file_path / f"{file_prefix}_test.csv")

    return train_df, valid_df, test_df


def gen_dataloaders(train_df, valid_df, test_df, input_col_names, target_col_names, batch_size=256):
    train_dataset = TensorDataset(torch.tensor(train_df[input_col_names].to_numpy(), dtype=torch.float32), 
              torch.tensor(train_df[target_col_names].to_numpy(), dtype=torch.float32))

    valid_dataset = TensorDataset(torch.tensor(valid_df[input_col_names].to_numpy(), dtype=torch.float32), 
                torch.tensor(valid_df[target_col_names].to_numpy(), dtype=torch.float32))

    test_dataset = TensorDataset(torch.tensor(test_df[input_col_names].to_numpy(), dtype=torch.float32), 
                torch.tensor(test_df[target_col_names].to_numpy(), dtype=torch.float32))
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, valid_dataloader, test_dataloader

if __name__ == '__main__':
    e = 0.003125 
    regions = [
        {   # Lid — only non-zero BC, most important
            'x_min': 0.0,     'x_max': 1,
            'y_min': 1 - e, 'y_max': 1,
            'dropout': 0.0,
            'random_state': 42,
            'groupby_cols': ['re', 'time']
        },
        {   # Bottom wall — no-slip
            'x_min': 0.0, 'x_max': 1,
            'y_min': 0.0, 'y_max': e,
            'dropout': 0.1,
            'random_state': 42,
            'groupby_cols': ['re', 'time']
        },
        {   # Left wall — no-slip, excludes corners already covered above/below
            'x_min': 0.0, 'x_max': e,
            'y_min': e,   'y_max': 1 - e,
            'dropout': 0.1,
            'random_state': 42,
            'groupby_cols': ['re', 'time']
        },
        {   # Right wall — no-slip
            'x_min': 1 - e, 'x_max': 1,
            'y_min': e,        'y_max': 1 - e,
            'dropout': 0.1,
            'random_state': 42,
            'groupby_cols': ['re', 'time']
        },
        {   # Interior — vortex core, high redundancy
            'x_min': e,       'x_max': 1 - e,
            'y_min': e,       'y_max': 1 - e,
            'dropout': 0.7,
            'random_state': 42,
            'groupby_cols': ['re', 'time']
        },
    ]
    print('Changing working dir...')
    os.chdir('/mnt2/ml_projekat/PINN-for-Simulating-2D-Incompressible-Fluid-Dynamics')  # Change working directory to the script's directory
    print('Working dir changed to:', os.getcwd())
    datapath = Path('data')
    files = ['data_train.csv', 'data_valid.csv']
    n_files = ['ndata_train.csv', 'ndata_valid.csv']

    for (file, n_file) in zip(files, n_files):
        print(f"Sampling {file} by regions and saving to {n_file}...")
        df = pd.read_csv(datapath / file)
        sampled_df = sample_by_region(df, regions)
        sampled_df.to_csv(datapath / n_file, index=False)



