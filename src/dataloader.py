import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd

from pathlib import Path


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