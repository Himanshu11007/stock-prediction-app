from typing import NamedTuple

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from features.engineer import compute_features, create_features


def show_candlestick_chart(data):
    fig = go.Figure(data=[
        go.Candlestick(
            x=data.index,
            open=data["Open"], high=data["High"],
            low=data["Low"],   close=data["Close"],
            name="Price",
        )
    ])
    fig.update_layout(
        title="Price Chart",
        xaxis_title="Date", yaxis_title="Price (₹)",
        height=420,
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9"),
    )
    st.plotly_chart(fig, width="stretch")


FEATURE_COLS = [
    "Close", "Volume", "Price_Change",
    "MA_5", "MA_10", "MA_Diff",
    "EMA_20", "EMA_50", "EMA_Cross", "Price_vs_EMA20",
    "RSI", "Momentum", "Volatility",
    "Volume_Change", "Volume_MA", "Volume_Ratio",
    "MACD", "MACD_Hist", "MACD_Cross",
    "BB_Width", "BB_Position",
    "ATR", "ATR_Pct",
    "ADX", "Plus_DI", "Minus_DI",
    "Vol_Breakout",
]


class InferenceData(NamedTuple):
    """Point-in-time split of a price history ending at bar D."""
    data:       pd.DataFrame          # features through D (last row = D); decision-engine input
    train_data: pd.DataFrame          # labelled rows only (== create_features output)
    X:          pd.DataFrame          # training features, rows <= D-1 (labels known at D)
    y:          pd.Series             # training labels
    y_train:    pd.Series             # first 80% of y — single-class guard, as prepare_data
    X_pred:     pd.DataFrame | None   # feature row of bar D; never part of X
    X_pred_invalid: tuple = ()        # bar-D features that are NaN/inf when X_pred is None


def prepare_inference_data(raw) -> InferenceData:
    """
    Production inference split. Training uses only rows whose label
    (Close[t+1] > Close[t]) is known; the prediction row is the latest bar D,
    whose label is unknown and which is therefore never trained on.

    X_pred is None when bar D has an incomplete feature row (too little
    history, or a NaN/inf feature); X_pred_invalid then names the offending
    columns. Callers must skip the prediction rather than fall back to an
    older row.
    """
    feats = compute_features(raw)
    train_data = feats.dropna()
    feature_cols = [c for c in FEATURE_COLS if c in feats.columns]

    X = train_data[feature_cols]
    y = train_data["Up"].astype(int)
    y_train = y[:int(len(X) * 0.8)]

    data = feats[feats.drop(columns="Up").notna().all(axis=1)]
    latest = raw.index[-1]
    X_pred = data.loc[[latest], feature_cols] if latest in data.index else None
    if X_pred is not None and latest in X.index:
        raise AssertionError("prediction row is part of the training set")

    invalid = ()
    if X_pred is None and latest in feats.index:
        row = feats.drop(columns="Up").loc[latest]
        invalid = tuple(row.index[row.isna()])

    return InferenceData(data, train_data, X, y, y_train, X_pred, invalid)


def prepare_data(data):
    data = create_features(data)

    # Only keep columns that exist after feature engineering
    feature_cols = [c for c in FEATURE_COLS if c in data.columns]

    X = data[feature_cols]
    y = data["Up"].astype(int)

    # Callers only read y_train (single-class guard); X_train/X_test/y_test
    # are unused but kept so the 7-tuple contract stays stable. Model
    # evaluation lives in models.trainer, not here.
    split   = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return data, X, y, X_train, X_test, y_train, y_test


def run_backtest(data, model, X):
    data = data.copy()   # avoid SettingWithCopyWarning
    data["Prediction"] = model.predict(X)
    proba = model.predict_proba(X)
    data["Confidence"] = proba.max(axis=1)
    data["Prediction"] = data["Prediction"].shift(1)

    data["Prediction"] = (
        (data["Prediction"] == 1) &
        (data["Confidence"] > 0.6) &
        (data["MA_5"] > data["MA_10"] * 1.01)
    ).astype(int)

    data["Market_Return"]   = data["Close"].pct_change()
    data["Strategy_Return"] = 0.0

    in_trade    = False
    entry_price = 0.0
    stop_loss   = -0.02
    take_profit =  0.05

    for i in range(1, len(data)):
        price = data["Close"].iloc[i]
        if not in_trade and data["Prediction"].iloc[i] == 1 and 40 < data["RSI"].iloc[i] < 70:
            in_trade    = True
            entry_price = price
        elif in_trade:
            trade_return = (price - entry_price) / entry_price
            if trade_return <= stop_loss or trade_return >= take_profit:
                data.loc[data.index[i], "Strategy_Return"] = trade_return
                in_trade = False

    data["Total_Return"]  = (1 + data["Strategy_Return"].fillna(0)).cumprod()
    data["Market_Total"]  = (1 + data["Market_Return"].fillna(0)).cumprod()
    return data.dropna()


def show_chart(data):
    st.subheader("💰 Strategy vs Buy & Hold")
    chart_data = data[["Total_Return", "Market_Total"]].copy()
    chart_data.columns = ["Model Strategy", "Buy & Hold"]
    st.line_chart(chart_data)


def show_metrics(data):
    final_value  = data["Total_Return"].iloc[-1]
    market_value = data["Market_Total"].iloc[-1]
    st.metric("Model result  (₹100 →)", f"₹{round(100 * final_value, 2)}")
    st.metric("Market result (₹100 →)", f"₹{round(100 * market_value, 2)}")


# ── Signal badge colours ────────────────────────────────────────────────────
_SIGNAL_STYLE = {
    "STRONG BUY":  {"bg": "#0a2e1a", "border": "#22c55e", "text": "#4ade80",  "icon": "🚀"},
    "BUY":         {"bg": "#0d2b1e", "border": "#238636", "text": "#3fb950",  "icon": "📈"},
    "HOLD":        {"bg": "#2b1d00", "border": "#bb8009", "text": "#d29922",  "icon": "⏸️"},
    "SELL":        {"bg": "#2d0c0c", "border": "#da3633", "text": "#f85149",  "icon": "📉"},
    "STRONG SELL": {"bg": "#1a0505", "border": "#b91c1c", "text": "#ef4444",  "icon": "🔥"},
}


def show_prediction(
    confidence: float,
    acc: float,
    model_name: str,
    final_signal: str,
    final_score: float,
    reason: str,
    factors: list[str] | None = None,
    risk: dict | None = None,
):
    st.subheader("📌 AI Signal")

    style = _SIGNAL_STYLE.get(final_signal, _SIGNAL_STYLE["HOLD"])
    score_pct = round(final_score * 100, 1)

    # ── Main signal card ──────────────────────────────────────────────────────
    st.markdown(f"""
<div style="background:{style['bg']};border:2px solid {style['border']};
            border-radius:12px;padding:1.2rem 1.4rem;margin-bottom:.8rem;">
  <div style="font-size:1.6rem;font-weight:900;color:{style['text']};letter-spacing:-.5px;">
    {style['icon']} {final_signal}
  </div>
  <div style="color:#c9d1d9;font-size:.85rem;margin-top:.4rem;">{reason}</div>
  <div style="display:flex;gap:1.5rem;margin-top:.9rem;flex-wrap:wrap;">
    <span style="color:#8b949e;font-size:.78rem;">
      Confluence&nbsp;<b style="color:{style['text']}">{score_pct}/100</b>
    </span>
    <span style="color:#8b949e;font-size:.78rem;">
      ML confidence&nbsp;<b style="color:#f0f6fc">{confidence:.0f}%</b>
    </span>
    <span style="color:#8b949e;font-size:.78rem;">
      Model accuracy&nbsp;<b style="color:#f0f6fc">{round(acc*100,1)}%</b>
    </span>
    <span style="color:#8b949e;font-size:.78rem;">
      Engine&nbsp;<b style="color:#f0f6fc">{model_name}</b>
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Confluence score bar ──────────────────────────────────────────────────
    st.progress(int(score_pct))

    # ── Factor breakdown ──────────────────────────────────────────────────────
    if factors:
        with st.expander("🔍 Why this signal? (factor breakdown)", expanded=True):
            for i, factor in enumerate(factors, 1):
                # Colour hint based on keywords
                if any(k in factor.lower() for k in ("bullish", "positive", "breakout", "above", "strong")):
                    colour = "#3fb950"
                elif any(k in factor.lower() for k in ("bearish", "negative", "below", "overbought", "weak")):
                    colour = "#f85149"
                else:
                    colour = "#8b949e"
                st.markdown(
                    f'<div style="padding:.25rem 0;font-size:.82rem;">'
                    f'<span style="color:{colour};font-weight:600">{i}.</span> '
                    f'<span style="color:#c9d1d9">{factor}</span></div>',
                    unsafe_allow_html=True,
                )

    # ── Risk panel ────────────────────────────────────────────────────────────
    if risk and risk.get("stop_loss") is not None:
        st.markdown("**⚠️ Risk levels (ATR-based)**")
        r1, r2, r3 = st.columns(3)
        r1.metric("Stop loss",  f"₹{risk['stop_loss']}", f"-{risk['risk_pct']}%")
        r2.metric("Target",     f"₹{risk['target']}",   f"+{risk['reward_pct']}%")
        r3.metric("R/R ratio",  f"{risk['rr_ratio']}:1")
        st.caption("⚠️ Not financial advice — these are model-generated levels only.")
