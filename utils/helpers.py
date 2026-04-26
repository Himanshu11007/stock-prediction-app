import streamlit as st
from features.engineer import(
    create_features
)
# -------------------------------
# HELPER FUNCTIONS
# -------------------------------

def prepare_data(data):
    """
    Create features and split data into train/test
    """
    # Create ML features
    data = create_features(data)

    # Select input features
    X = data[["Close", "Volume", "Price_Change", "MA_5", "MA_10", "RSI"]]

    # Target column (Up/Down)
    y = data["Up"]

    # Time-based split (important for stock data)
    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return data, X, y, X_train, X_test, y_train, y_test


def run_backtest(data, model, X):
    """
    Simulate trading strategy based on model predictions
    """
    # Predict for full dataset
    data["Prediction"] = model.predict(X)

    # Shift prediction (avoid future leakage)
    data["Prediction"] = data["Prediction"].shift(1)

    # Market return (daily % change)
    data["Market_Return"] = data["Close"].pct_change()

    # Strategy return (only when model says BUY)
    data["Strategy_Return"] = data["Prediction"] * data["Market_Return"]

    # Portfolio growth (model strategy)
    data["Total_Return"] = (1 + data["Strategy_Return"].fillna(0)).cumprod()

    # Market growth (buy & hold)
    data["Market_Total"] = (1 + data["Market_Return"].fillna(0)).cumprod()

    # Clean data
    data = data.dropna()

    return data


def show_chart(data):
    """
    Display charts in UI
    """
    st.subheader("💰 Investment Growth (Model vs Market)")

    # Rename for user clarity
    chart_data = data[["Total_Return", "Market_Total"]].copy()
    chart_data.columns = ["Model Strategy", "Buy & Hold"]

    st.line_chart(chart_data)


def show_metrics(data):
    """
    Show final investment values
    """
    final_value = data["Total_Return"].iloc[-1]
    market_value = data["Market_Total"].iloc[-1]

    st.metric("Model Result (₹100 →)", f"₹{round(100 * final_value, 2)}")
    st.metric("Market Result (₹100 →)", f"₹{round(100 * market_value, 2)}")


def show_prediction(pred, confidence, acc, name):
    """
    Show BUY/SELL signal UI
    """
    st.subheader("📌 Trading Signal")

    if pred[0] == 1:
        st.markdown(f"""
        <div style="background-color:#d4edda;padding:20px;border-radius:10px;border-left:6px solid green;">
        <h2>📈 BUY Signal</h2>
        <p><b>Confidence:</b> {confidence}%</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color:#f8d7da;padding:20px;border-radius:10px;border-left:6px solid red;">
        <h2>📉 SELL Signal</h2>
        <p><b>Confidence:</b> {confidence}%</p>
        </div>
        """, unsafe_allow_html=True)

    # Show model info
    st.metric("Model Accuracy", f"{round(acc * 100, 2)}%")
    st.caption(f"Model Used: {name}")
