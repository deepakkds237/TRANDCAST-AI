# TECHNICAL INDICATORS SECTION
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from forecast import make_forecast
import subprocess, sys
import matplotlib.pyplot as plt

st.title("🌟 TrendCast AI – Smart Forecasting App")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

data_path = Path("data")
data_path.mkdir(exist_ok=True)

# SESSION STATE
if "csv_loaded" not in st.session_state:
    st.session_state.csv_loaded = False

if "model_trained" not in st.session_state:
    st.session_state.model_trained = False


# ------------------------------- INDICATOR FUNCTIONS -------------------------------
def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_Bollinger(series, period=20):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    return sma, upper, lower


def compute_supertrend(df, period=7, multiplier=3):
    hl2 = (df['High'] + df['Low']) / 2
    atr = (df['High'] - df['Low']).rolling(period).mean()

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(index=df.index)

    for i in range(len(df)):
        if i == 0:
            supertrend.iloc[i] = upper_band.iloc[i]
        else:
            if df['Close'].iloc[i] > upper_band.iloc[i - 1]:
                supertrend.iloc[i] = lower_band.iloc[i]
            elif df['Close'].iloc[i] < lower_band.iloc[i - 1]:
                supertrend.iloc[i] = upper_band.iloc[i]
            else:
                supertrend.iloc[i] = supertrend.iloc[i - 1]

    return supertrend


# ------------------------------------------------------------ LOAD CSV ------------------------------------------------------------
if uploaded:
    save_path = data_path / "uploaded.csv"
    uploaded.seek(0)
    df = pd.read_csv(uploaded)

    df.columns = df.columns.str.strip()

    st.session_state.csv_loaded = True
    st.session_state.save_path = save_path
    st.session_state.df = df

    df.to_csv(save_path, index=False)


# ------------------------------------------------------------ UI + INDICATORS ------------------------------------------------------------
if st.session_state.csv_loaded:
    df = st.session_state.df

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())


    # -------------------------- INDICATORS --------------------------
    st.subheader("📊 Technical Indicators")

    if {'Close', 'High', 'Low'}.issubset(df.columns):

        # Convert numeric columns safely
        df['Close'] = pd.to_numeric(df['Close'], errors="coerce")
        df['High'] = pd.to_numeric(df['High'], errors="coerce")
        df['Low'] = pd.to_numeric(df['Low'], errors="coerce")

        df.dropna(subset=['Close','High','Low'], inplace=True)

        df['RSI'] = compute_RSI(df['Close'])
        df['BB_MID'], df['BB_UPPER'], df['BB_LOWER'] = compute_Bollinger(df['Close'])
        df['Supertrend'] = compute_supertrend(df)

        st.success("Indicators Added Successfully!")

        # ------ LINE PLOT ------
        st.subheader("📈 Line Chart")
        fig1, ax1 = plt.subplots(figsize=(8, 3))
        ax1.plot(df['Close'], label="Close Price")
        ax1.set_title("Line Chart")
        ax1.legend()
        st.pyplot(fig1)
        plt.close(fig1)

        # ------ SCATTER PLOT ------
        st.subheader("📌 Scatter Plot (Close vs RSI)")
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.scatter(df['Close'], df['RSI'])
        ax2.set_title("Scatter: Close vs RSI")
        st.pyplot(fig2)
        plt.close(fig2)

        # ------ BOLLINGER ------
        st.subheader("📉 Bollinger Bands")
        fig3, ax3 = plt.subplots(figsize=(8, 3))
        ax3.plot(df['Close'], label="Close")
        ax3.plot(df['BB_UPPER'], label="Upper Band")
        ax3.plot(df['BB_LOWER'], label="Lower Band")
        ax3.legend()
        st.pyplot(fig3)
        plt.close(fig3)

    else:
        st.warning("Need columns: Close, High, Low to compute indicators.")


    # ----------------------- FORECAST COLUMN SELECT -----------------------
    st.subheader("🎯 Select Forecast Column")

    # Only numeric columns allowed
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    column_to_forecast = st.selectbox(
        "Select Column to Forecast (Numbers Only)",
        numeric_cols
    )


    # ----------------------- TRAIN MODEL -----------------------
    if st.button("Train Model"):
        try:
            subprocess.check_call([sys.executable, "src/train_lstm.py", column_to_forecast])
            st.session_state.model_trained = True
            st.success("Model Training Completed! 🎉")
        except Exception as e:
            st.error(f"Training Failed: {e}")


    # ----------------------- FORECAST -----------------------
    if st.session_state.model_trained:
        if st.button("Forecast Next Days"):
            try:
                preds = make_forecast(
                    csv_path=str(st.session_state.save_path),
                    column=column_to_forecast,
                    seq_len=10,
                    days=5
                )

                st.subheader("🔮 Forecasted Values")
                st.write(preds)

                # Forecast Plot
                st.subheader("📈 Forecast Plot")

                last_actual = df[column_to_forecast].dropna().values[-20:]
                forecast_points = preds

                t1 = np.arange(len(last_actual))
                t2 = np.arange(len(last_actual), len(last_actual) + len(forecast_points))

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(t1, last_actual, label="Actual (Last 20)")
                ax.plot(t2, forecast_points, label="Forecast", linestyle='--')

                ax.set_title("Actual vs Forecast")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

            except Exception as e:
                st.error(f"Error: {e}")

else:
    st.info("📤 Please upload a CSV file.")
