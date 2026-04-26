import streamlit as st
from data.loader import load_data
from features.engineer import create_features
from models.trainer import train_model
from utils.helpers import(
    prepare_data,run_backtest,show_chart,show_metrics,show_prediction
)

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(page_title="Stock Predictor", layout="wide")

# App header
st.caption("Note: This is an experimental ML model. Not financial advice")
st.title("📈 Stock Prediction App")
st.markdown("Predict next-day stock movement using machine learning")


# -------------------------------
# USER INPUT
# -------------------------------
stock_name = st.selectbox(
    "Select Stock",
    ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
)


# -------------------------------
# MAIN EXECUTION
# -------------------------------

if st.button("Predict"):

    # Show loading spinner
    with st.spinner("Fetching and analyzing data..."):
        data = load_data(stock_name)

    # Handle empty data
    if data.empty:
        st.error("Invalid stock name or no data found")
        st.stop()

    # Prepare data
    data, X, y, X_train, X_test, y_train, y_test = prepare_data(data)

    # Train model
    model, acc, name = train_model(X_train, y_train, X_test, y_test)

    # Latest prediction
    latest = X.tail(1)
    pred = model.predict(latest)

    # Prediction confidence
    prob = model.predict_proba(latest)
    confidence = round(max(prob[0]) * 100, 2)

    # Run backtesting
    data = run_backtest(data, model, X)

    # Layout
    col1, col2 = st.columns(2)

    # LEFT → Charts + metrics
    with col1:
        show_chart(data)
        show_metrics(data)
        st.write("Latest Close Price:", round(data["Close"].iloc[-1], 2))

    # RIGHT → Prediction
    with col2:
        show_prediction(pred, confidence, acc, name)