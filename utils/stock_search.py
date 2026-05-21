import pandas as pd
from pathlib import Path

def load_stock_data():
    csv_path = Path(__file__).parent.parent / "data" / "nifty50.csv"
    return pd.read_csv(csv_path)