import numpy as np
import pandas as pd
import torch

from .config import (
    EPOCHS,
    HIDDEN_DIM,
    INPUT_DIM,
    LR,
    SEED,
    SEQ_LENGTH,
    THRESHOLD_PERCENTILE,
)
from .model import Autoencoder
from .utils import (
    LossFunctionType,
    create_sequences,
    get_loss_function,
    set_seeds,
)


def detect_anomalies_with_lstm_autoencoder(
    df: pd.DataFrame,
    part_name: str,
    seq_length: int = SEQ_LENGTH,
    hidden_dim: int = HIDDEN_DIM,
    epochs: int = EPOCHS,
    lr: float = LR,
    loss_type: LossFunctionType = LossFunctionType.MSE,
    verbose: bool = False,
    threshold_percentile: int = THRESHOLD_PERCENTILE,
    seed: int = SEED,
):
    set_seeds(seed)

    part_data = df[[f"{part_name}_x", f"{part_name}_y"]].values

    part_sequences = create_sequences(part_data, seq_length)
    part_sequences_tensor = torch.FloatTensor(part_sequences)

    model = Autoencoder(INPUT_DIM, hidden_dim, seq_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = get_loss_function(loss_type)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(part_sequences_tensor)
        loss = criterion(outputs, part_sequences_tensor)
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        reconstructed = model(part_sequences_tensor)
        reconstruction_errors = torch.mean(
            torch.square(part_sequences_tensor - reconstructed), dim=[1, 2]
        )

    threshold = np.percentile(reconstruction_errors, threshold_percentile)
    anomaly_indices = np.where(reconstruction_errors > threshold)[0]

    return anomaly_indices
