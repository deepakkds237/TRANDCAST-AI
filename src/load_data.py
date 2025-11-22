import pandas as pd

def load_csv(path):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Unable to read CSV file: {e}")

    # Clean column names (remove spaces & quotes)
    df.columns = (
        df.columns
        .str.strip()
        .str.replace("'", "", regex=False)
        .str.replace('"', "", regex=False)
    )

    # Drop completely empty columns
    df.dropna(axis=1, how="all", inplace=True)

    if df.empty:
        raise ValueError("CSV file contains no usable data")

    return df
