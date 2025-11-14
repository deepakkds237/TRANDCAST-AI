import numpy as np
import os
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from load_data import load_csv
from preprocess import create_dataset, scale_series

def train(csv_path="data/sample.csv", seq_len=10, epochs=5, batch_size=8):
    df = load_csv(csv_path)
    close = df["Close"].values.astype(float)

    # Scale data
    scaled, scaler = scale_series(close)

    # Create sequence dataset
    X, y = create_dataset(scaled, seq_len)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # LSTM model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    model.fit(X, y, epochs=epochs, batch_size=batch_size)

    # Save model + scaler
    os.makedirs("models", exist_ok=True)
    model.save("models/trendcast_model.h5")
    joblib.dump(scaler, "models/scaler.pkl")

    print("Model training complete! Saved model + scaler.")

if __name__ == "__main__":
    train()
