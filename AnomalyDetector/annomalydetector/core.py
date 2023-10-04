import numpy as np
import torch

from .model import Autoencoder
from .utils import (
    LossFunctionType,
    create_sequences,
    get_loss_function,
    set_seeds,
)


def detect_anomalies_with_lstm_autoencoder(
    part_data: np.ndarray,
    seq_length: int = 10,
    hidden_dim: int = 30,
    epochs: int = 200,
    lr: float = 1e-3,
    loss_type: LossFunctionType = LossFunctionType.MSE,
    verbose: bool = False,
    threshold_percentile: int = 95,
    seed: int = 42,
):
    set_seeds(seed)

    _, input_dim = part_data.shape

    part_sequences = create_sequences(part_data, seq_length)
    part_sequences_tensor = torch.FloatTensor(part_sequences)

    model = Autoencoder(input_dim, hidden_dim, seq_length)
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

    return anomaly_indices, reconstruction_errors


def detect_anomalies_with_moving_avg_std(
    data: np.ndarray,
    window_size: int = 10,
    threshold_factor: float = 2.5,
):
    num_dimensions = data.shape[1]
    squared_residuals_per_dimension = []

    for dim in range(num_dimensions):
        dim_data = data[:, dim]

        moving_avg = np.convolve(
            dim_data, np.ones(window_size) / window_size, mode="valid"
        )

        squared_residuals = (
            dim_data[window_size - 1 :] - moving_avg  # noqa
        ) ** 2
        squared_residuals_per_dimension.append(squared_residuals)

    distances = np.sqrt(
        np.sum(np.array(squared_residuals_per_dimension), axis=0)
    )

    moving_avg_distance = np.convolve(
        distances, np.ones(window_size) / window_size, mode="valid"
    )

    moving_std_distance = [
        np.std(distances[i - window_size : i])  # noqa
        for i in range(window_size, len(distances) + 1)
    ]

    upper_bound = moving_avg_distance + threshold_factor * np.array(
        moving_std_distance
    )

    anomaly_indices = (
        np.where(distances[window_size - 1 :] > upper_bound)[0]  # noqa
        + window_size
        - 1
    )

    return anomaly_indices
