import streamlit as st
import pandas as pd
from pathlib import Path
from forecast import make_forecast

st.title("🌟 TrendCast AI – Smart Forecasting App")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
data_path = Path("data")
data_path.mkdir(exist_ok=True)

if uploaded:
    save_path = data_path / "uploaded.csv"
    with open(save_path, "wb") as f:
        f.write(uploaded.getbuffer())

    df = pd.read_csv(save_path)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Let user select column to forecast
    column_to_forecast = st.selectbox("Select Column to Forecast", df.columns)

    if st.button("Train Model"):
        import subprocess, sys
        subprocess.check_call([sys.executable, "src/train_lstm.py"])
        st.success("Training Completed")

    if st.button("Forecast Next Days"):
        try:
            preds = make_forecast(csv_path=str(save_path), column=column_to_forecast, seq_len=10, days=5)
            st.subheader("Forecasted Values")
            st.write(preds)
        except Exception as e:
            st.error(f"Error: {e}")

else:
    st.info("Please upload a CSV file.")
