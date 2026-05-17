import pandas as pd

def load_stock_data():
    return pd.read_csv(
        "data/nifty50.csv"
)