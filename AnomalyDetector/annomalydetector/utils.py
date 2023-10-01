from enum import Enum

import numpy as np
import torch
import torch.nn as nn


def create_sequences(data: np.ndarray, seq_length: int):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i : i + seq_length]  # noqa: E203
        sequences.append(seq)
    return np.array(sequences)


class LossFunctionType(Enum):
    MSE = "MSE"
    MAE = "MAE"
    HUBER = "Huber"


def get_loss_function(loss_type):
    if loss_type == LossFunctionType.MSE:
        return nn.MSELoss()
    elif loss_type == LossFunctionType.MAE:
        return nn.L1Loss()
    elif loss_type == LossFunctionType.HUBER:
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unsupported loss function type: {loss_type}")


def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
