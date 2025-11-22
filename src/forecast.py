import numpy as np
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from load_data import load_csv
from preprocess import scale_series


# ------------------------------
# 🔹 RSI Function
# ------------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ------------------------------
# 🔹 Bollinger Bands Function
# ------------------------------
def compute_bollinger(series, window=20):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()

    upper = ma + (2 * std)
    lower = ma - (2 * std)
    return ma, upper, lower


# ------------------------------
# 🔹 Supertrend Function
# ------------------------------
def compute_supertrend(df, period=10, multiplier=3):
    hl2 = (df['High'] + df['Low']) / 2

    atr = df['High'].rolling(period).max() - df['Low'].rolling(period).min()
    atr = atr.rolling(period).mean()

    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    supertrend = pd.Series(index=df.index, dtype=float)

    for i in range(len(df)):
        if i == 0:
            supertrend.iloc[i] = upperband.iloc[i]
        else:
            if df['Close'].iloc[i] > supertrend.iloc[i - 1]:
                supertrend.iloc[i] = lowerband.iloc[i]
            else:
                supertrend.iloc[i] = upperband.iloc[i]

    return supertrend


# ------------------------------
# 🔵 MAIN FORECAST FUNCTION (UPDATED)
# ------------------------------
def make_forecast(csv_path="data/uploaded.csv", column="Close", seq_len=10, days=5):
    # Load CSV
    df = load_csv(csv_path)

    # Column validation
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found. Available columns: {list(df.columns)}"
        )

    close = df[column].astype(float).values

    if len(close) < seq_len:
        raise ValueError("Dataset is too short to forecast.")

    # Load model
    model_path = "models/trendcast_model.keras"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model not found! Train the model first to create trendcast_model.keras"
        )

    model = load_model(model_path)
    scaler = joblib.load("models/scaler.pkl")

    # Scale series
    scaled, _ = scale_series(close, scaler=scaler)

    x_input = scaled[-seq_len:].reshape(1, seq_len, 1)
    preds_scaled = []

    # Multi-step prediction
    for _ in range(days):
        pred = model.predict(x_input, verbose=0)[0][0]
        preds_scaled.append(pred)
        x_input = np.append(x_input[:, 1:, :], [[[pred]]], axis=1)

    final_pred = scaler.inverse_transform(
        np.array(preds_scaled).reshape(-1, 1)
    ).flatten()

    # -----------------------------------------------------
    # 🔵 INDICATORS CALCULATION
    # -----------------------------------------------------
    df["RSI"] = compute_rsi(df["Close"])
    df["MA20"], df["BB_UP"], df["BB_LOW"] = compute_bollinger(df["Close"])
    df["Supertrend"] = compute_supertrend(df)

    # -----------------------------------------------------
    # 🔵 LINE PLOT
    # -----------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(df["Close"], label="Close Price")
    plt.plot(df["MA20"], label="MA20")
    plt.plot(df["BB_UP"], label="Upper Band")
    plt.plot(df["BB_LOW"], label="Lower Band")
    plt.plot(df["Supertrend"], label="Supertrend")
    plt.title("Price + Indicators")
    plt.legend()
    plt.show()

    # -----------------------------------------------------
    # 🔵 SCATTER PLOT (RSI)
    # -----------------------------------------------------
    plt.figure(figsize=(12, 5))
    plt.scatter(df.index, df["RSI"], s=10)
    plt.title("RSI Scatter Plot")
    plt.show()

    return final_pred


if __name__ == "__main__":
    print(make_forecast())
