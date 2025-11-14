import numpy as np
import joblib
from tensorflow.keras.models import load_model
from load_data import load_csv
from preprocess import scale_series

def make_forecast(csv_path="data/sample.csv", column="Close", seq_len=10, days=5):
    df = load_csv(csv_path)

    if column not in df.columns:
        raise ValueError(f"CSV must contain a '{column}' column. Available columns: {list(df.columns)}")

    close = df[column].values.astype(float)

    # Load scaler + model
    scaler = joblib.load("models/scaler.pkl")
    model = load_model("models/trendcast_model.h5")

    # Scale series with scaler
    scaled, _ = scale_series(close, scaler=scaler)

    # Prepare last window of data
    x_input = scaled[-seq_len:].reshape(1, seq_len, 1)

    preds_scaled = []
    for i in range(days):
        pred = model.predict(x_input, verbose=0)[0][0]
        preds_scaled.append(pred)
        x_input = np.append(x_input[:, 1:, :], [[[pred]]], axis=1)

    # Convert back to original values
    final_pred = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    return final_pred

if __name__ == "__main__":
    print(make_forecast())
