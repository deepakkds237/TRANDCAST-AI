import numpy as np
import os
import sys
import joblib
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from load_data import load_csv
from preprocess import create_dataset, scale_series


def train(csv_path="data/uploaded.csv", column="Close", seq_len=10, epochs=20):
    # -----------------------------------
    # 1️⃣ Load CSV
    # -----------------------------------
    df = load_csv(csv_path)

    # Clean column name
    column = column.strip()

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available columns: {list(df.columns)}")

    # Extract series
    series = df[column].astype(float).values

    if len(series) <= seq_len:
        raise ValueError(f"Not enough rows. Need at least {seq_len + 1} values.")

    # -----------------------------------
    # 2️⃣ Scale values
    # -----------------------------------
    scaled, scaler = scale_series(series)

    # Prepare dataset
    X, y = create_dataset(scaled, seq_len)

    if len(X) == 0:
        raise ValueError("Dataset is too short after scaling/cleaning.")

    # LSTM reshape
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # -----------------------------------
    # 3️⃣ Build & Train Model
    # -----------------------------------
    model = Sequential([
        Input(shape=(seq_len, 1)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    print("🚀 Training started...")
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=1)
    print("🎉 Training finished.")

    # -----------------------------------
    # 4️⃣ Save Model + Scaler
    # -----------------------------------
    os.makedirs("models", exist_ok=True)

    model.save("models/trendcast_model.keras")
    joblib.dump(scaler, "models/scaler.pkl")

    print("\n✅ Training complete! Model saved as 'models/trendcast_model.keras'")
    print("📦 Scaler saved as 'models/scaler.pkl'")



if __name__ == "__main__":
    column_name = "Close"

    if len(sys.argv) > 1:
        column_name = sys.argv[1]

    train(column=column_name)
