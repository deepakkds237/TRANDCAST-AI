🌟 TrendCast-AI – Smart Forecasting App

TrendCast-AI is an intelligent forecasting application that predicts future trends of financial or time-series data using LSTM (Long Short-Term Memory) neural networks. Users can upload CSV files, train the model, and generate forecasts for selected columns with ease.

🚀 Features

Upload your own CSV datasets for analysis.

Interactive data preview with Streamlit.

Train LSTM models directly from the app.

Forecast future values for selected columns.

Scalable and easy-to-use with support for multiple datasets.

Automatic handling of data preprocessing and scaling.

🛠️ Technology Stack

Python 3.10+

Streamlit – Frontend UI

TensorFlow / Keras – LSTM model

Plotly – Data visualization

Pandas & NumPy – Data manipulation

Joblib – Saving/loading scalers

Pathlib – File handling

📁 Project Structure
trendcast-ai/
│
├── data/                  # Folder for uploaded CSVs
├── models/                # Saved scalers and trained models
│   ├── scaler.pkl
│   └── trendcast_model.h5
├── src/                   # Source code
│   ├── app.py             # Streamlit main app
│   ├── forecast.py        # Forecasting functions
│   ├── train_lstm.py      # Script to train LSTM model
│   ├── load_data.py       # CSV loading utility
│   └── preprocess.py      # Data scaling and preprocessing
├── venv/                  # Python virtual environment
└── README.md              # Project documentation

⚡ Installation

Clone the repository:

git clone https://github.com/yourusername/trendcast-ai.git
cd trendcast-ai


Create a virtual environment:

python -m venv venv


Activate the virtual environment:

Windows:

venv\Scripts\activate


Linux/Mac:

source venv/bin/activate


Install dependencies:

pip install -r requirements.txt

🏃 How to Run

Launch the Streamlit app:

streamlit run src/app.py


Upload a CSV file containing your time-series data.

Preview the data and select the column you want to forecast.

Train the model (optional) or directly forecast next values.

View predictions in the interactive dashboard.

🔧 Usage Example
from forecast import make_forecast

predictions = make_forecast(
    csv_path="data/uploaded.csv",
    column="Close",
    seq_len=10,
    days=5
)
print(predictions)

💡 Notes

Ensure your CSV has numeric columns for forecasting.

Avoid extremely large seq_len or days values to prevent memory issues.

For best results, normalize your data if not using the built-in scaler.

📈 Future Improvements

Auto-detect numeric columns for forecasting.

Option to visualize predictions with Plotly charts.

Add support for multi-step and multi-column forecasting.

Integration with real-time financial APIs.
