import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_dataset(series, seq_len):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    return np.array(X), np.array(y)

def scale_series(series, scaler=None):
    series = series.reshape(-1, 1)

    if scaler is None:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series)
    else:
        scaled = scaler.transform(series)

    return scaled, scaler
