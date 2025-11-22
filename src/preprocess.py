import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_dataset(series, seq_len):
    """
    Convert time-series into supervised learning format.
    """
    series = np.array(series).astype(float)

    if len(series) <= seq_len:
        raise ValueError(f"Series too short. Need > {seq_len} values.")

    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])

    return np.array(X), np.array(y)


def scale_series(series, scaler=None):
    """
    Scale a 1D series using MinMaxScaler.
    """
    series = np.array(series).astype(float).reshape(-1, 1)

    # Remove NaN and Inf automatically
    if np.isnan(series).any() or np.isinf(series).any():
        raise ValueError("Series contains NaN or infinite values.")

    if scaler is None:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series)
    else:
        scaled = scaler.transform(series)

    return scaled, scaler
