import numpy as np


def create_features(data):
    # ── Target ────────────────────────────────────────────────────────────────
    data["Up"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

    # ── Momentum & trend ──────────────────────────────────────────────────────
    data["Price_Change"] = data["Close"].pct_change()
    data["MA_5"]         = data["Close"].rolling(5).mean()
    data["MA_10"]        = data["Close"].rolling(10).mean()
    data["MA_Diff"]      = data["MA_5"] - data["MA_10"]
    data["Momentum"]     = data["Close"] - data["Close"].shift(5)
    data["Volatility"]   = data["Close"].rolling(10).std()

    # ── Volume ────────────────────────────────────────────────────────────────
    data["Volume_Change"] = data["Volume"].pct_change()
    data["Volume_MA"]     = data["Volume"].rolling(5).mean()

    # ── RSI (14-period) ───────────────────────────────────────────────────────
    delta    = data["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs           = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI"]  = (100 - 100 / (1 + rs)).fillna(100)

    # ── MACD (12 / 26 / 9) ───────────────────────────────────────────────────
    ema12               = data["Close"].ewm(span=12, adjust=False).mean()
    ema26               = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"]        = ema12 - ema26
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Hist"]   = data["MACD"] - data["MACD_Signal"]

    # ── Bollinger Bands (20-period) ───────────────────────────────────────────
    bb_ma  = data["Close"].rolling(20).mean()
    bb_std = data["Close"].rolling(20).std()
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    data["BB_Width"]    = (bb_upper - bb_lower) / bb_ma
    data["BB_Position"] = (
        (data["Close"] - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    )

    data = data.replace([np.inf, -np.inf], np.nan)
    return data.dropna()
