import yfinance as yf

def load_data(symbol):
    data = yf.download(symbol,start="2022-01-01")
    data.columns = data.columns.get_level_values(0)
    return data