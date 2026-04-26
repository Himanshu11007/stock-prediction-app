def create_features(data):
        data["Up"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

        # Price change momentum
        data["Price_Change"] = data["Close"].pct_change()

        # Moving averages (trend indicators)
        data["MA_5"] = data["Close"].rolling(window=5).mean()
        data["MA_10"] = data["Close"].rolling(window=10).mean()

        # RSI (Relative Strength Index)
        delta = data["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / avg_loss
        data["RSI"] = 100 - (100 / (1 + rs))

        return data.dropna()