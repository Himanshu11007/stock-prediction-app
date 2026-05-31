import numpy as np
import pandas as pd


def create_features(data):
    data = data.copy()   # avoid SettingWithCopyWarning on slice inputs

    # ── Target ────────────────────────────────────────────────────────────────
    data["Up"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

    # ── Price change & momentum ───────────────────────────────────────────────
    data["Price_Change"] = data["Close"].pct_change()
    data["Momentum"]     = data["Close"] - data["Close"].shift(5)

    # ── Moving averages ───────────────────────────────────────────────────────
    data["MA_5"]   = data["Close"].rolling(5).mean()
    data["MA_10"]  = data["Close"].rolling(10).mean()
    data["MA_Diff"] = data["MA_5"] - data["MA_10"]
    data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()
    data["EMA_50"] = data["Close"].ewm(span=50, adjust=False).mean()
    # +1 if EMA20 > EMA50 (golden cross zone), -1 otherwise
    data["EMA_Cross"] = np.where(data["EMA_20"] > data["EMA_50"], 1.0, -1.0)
    # Price distance from EMA20 as % (positive = above EMA)
    data["Price_vs_EMA20"] = (data["Close"] - data["EMA_20"]) / data["EMA_20"]

    # ── Volatility ────────────────────────────────────────────────────────────
    data["Volatility"] = data["Close"].rolling(10).std()

    # ── Volume ────────────────────────────────────────────────────────────────
    data["Volume_Change"] = data["Volume"].pct_change()
    data["Volume_MA"]     = data["Volume"].rolling(20).mean()
    data["Volume_Ratio"]  = data["Volume"] / data["Volume_MA"].replace(0, np.nan)

    # ── RSI (14) ──────────────────────────────────────────────────────────────
    delta    = data["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()   # Wilder smoothing
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs           = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI"]  = (100 - 100 / (1 + rs)).fillna(50)

    # ── MACD (12 / 26 / 9) ───────────────────────────────────────────────────
    ema12               = data["Close"].ewm(span=12, adjust=False).mean()
    ema26               = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"]        = ema12 - ema26
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Hist"]   = data["MACD"] - data["MACD_Signal"]
    # Bullish crossover: histogram just turned positive
    data["MACD_Cross"] = np.where(
        (data["MACD_Hist"] > 0) & (data["MACD_Hist"].shift(1) <= 0), 1.0,
        np.where(
            (data["MACD_Hist"] < 0) & (data["MACD_Hist"].shift(1) >= 0), -1.0,
            0.0,
        ),
    )

    # ── Bollinger Bands (20) ──────────────────────────────────────────────────
    bb_ma    = data["Close"].rolling(20).mean()
    bb_std   = data["Close"].rolling(20).std()
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    data["BB_Width"]    = (bb_upper - bb_lower) / bb_ma.replace(0, np.nan)
    data["BB_Position"] = (
        (data["Close"] - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    )

    # ── ATR (14) — Average True Range ────────────────────────────────────────
    high_low   = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift(1)).abs()
    low_close  = (data["Low"]  - data["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data["ATR"] = true_range.ewm(com=13, adjust=False).mean()   # Wilder ATR
    data["ATR_Pct"] = data["ATR"] / data["Close"]               # normalised

    # ── ADX (14) — Trend strength ─────────────────────────────────────────────
    up_move   = data["High"].diff()
    down_move = -data["Low"].diff()
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm  = pd.Series(plus_dm,  index=data.index)
    minus_dm = pd.Series(minus_dm, index=data.index)
    atr14 = true_range.ewm(com=13,adjust=False).mean()
    plus_di14 = (100 *plus_dm.ewm(com=13, adjust=False).mean() / atr14.replace(0, np.nan))

    minus_di14 = (
        100 *
        minus_dm.ewm(com=13, adjust=False).mean()
        / atr14.replace(0, np.nan)
    )

    dx = (
        100 *
        (plus_di14 - minus_di14).abs()
        / (plus_di14 + minus_di14).replace(0, np.nan)
    )

    data["ADX"]      = dx.ewm(com=13, adjust=False).mean()
    data["ADX"]      = data["ADX"].fillna(0)
    data["Plus_DI"]  = plus_di14
    data["Minus_DI"] = minus_di14

    # ── Volume breakout flag ──────────────────────────────────────────────────
    # 1.0 when current volume > 1.5× 20-day avg AND price moved up
    data["Vol_Breakout"] = np.where(
        (data["Volume_Ratio"] > 1.5) & (data["Price_Change"] > 0), 1.0, 0.0
    )

    data = data.replace([np.inf, -np.inf], np.nan)
    return data.dropna()


def get_trend_signal(data):

    try:

        if data is None or len(data) < 60:
            return {
                "trend": "SIDEWAYS",
                "score": 0
            }

        df = create_features(data.copy())

        latest = df.iloc[-1]

        ema20 = latest["EMA_20"]
        ema50 = latest["EMA_50"]
        rsi = latest["RSI"]
        macd = latest["MACD"]
        macd_signal = latest["MACD_Signal"]

        score = 0

        # EMA trend
        ema_diff_pct = ((ema20 - ema50) / ema50) * 100

        if ema_diff_pct > 1:
            score += 1
        elif ema_diff_pct < -1:
            score -= 1

        # RSI momentum
        if rsi > 60:
            score += 1
        elif rsi < 40:
            score -= 1

        # MACD momentum
        if macd > macd_signal:
            score += 1
        elif macd < macd_signal:
            score -= 1

        normalized_score = round(score / 3, 2)

        # Final trend classification
        if score >= 2:

            return {
                "trend": "BULLISH",
                "score": normalized_score
            }

        elif score <= -2:

            return {
                "trend": "BEARISH",
                "score": normalized_score
            }

        else:

            return {
                "trend": "SIDEWAYS",
                "score": normalized_score
            }

    except Exception as e:

        import traceback
        traceback.print_exc()

        print(f"Trend detection error: {e}")

        return {
            "trend": "SIDEWAYS",
            "score": 0
        }


