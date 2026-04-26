import numpy as np
def create_features(data):
        data["Up"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

        # Price change momentum
        data["Price_Change"] = data["Close"].pct_change()

        # Moving averages (trend indicators)
        data["MA_5"] = data["Close"].rolling(window=5).mean()
        data["MA_10"] = data["Close"].rolling(window=10).mean()
        data["Momentum"] = data["Close"] - data["Close"].shift(5)
        data["Volatility"] = data["Close"].rolling(10).std()
        data["Volume_Change"] = data["Volume"].pct_change()
        data["Volume_MA"] =data["Volume"].rolling(5).mean()
        data["MA_Diff"] = data["MA_5"] - data["MA_10"]

        # RSI (Relative Strength Index)
        delta = data["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / avg_loss
        data["RSI"] = 100 - (100 / (1 + rs))

        data = data.replace([np.inf,-np.inf],np.nan)
        return data.dropna()