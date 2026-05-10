import yfinance as yf

def load_data(stock_name):
    data = yf.download(stock_name,period="5y")

    data.columns =[col[0]if isinstance(col,tuple) else col for col in data.columns]
    return data