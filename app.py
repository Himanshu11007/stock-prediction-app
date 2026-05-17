import streamlit as st
import pandas as pd

# ---------------------------------
# IMPORTS
# ---------------------------------

# Data
from data.loader import load_data

# Model
from models.trainer import train_model

# News
from news.api import fetch_news
from news.sentiment import analyze_overall_sentiment

# Utils
from utils.helpers import (
    prepare_data,
    run_backtest,
    show_chart,
    show_metrics,
    show_prediction,
    show_candlestick_chart
)
from utils.stock_search import load_stock_data
from utils.decision_engine import generate_signal
from utils.recommendation_engine import get_top_recommendations


# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="Stock Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------
# HEADER
# ---------------------------------

st.caption("⚠️ Note: This is an experimental ML model. Not financial advice.")
st.title("📈 Stock Prediction App")
st.markdown("Predict next-day stock movement using machine learning & sentiment analysis")


# ---------------------------------
# LOAD STOCK DATABASE
# ---------------------------------

try:
    stocks_df = load_stock_data()
    stocks_df.columns = stocks_df.columns.str.strip()
except Exception as e:
    st.error(f"Failed to load stock database: {e}")
    st.stop()


# ---------------------------------
# STOCK SELECTION
# ---------------------------------

if stocks_df.empty:
    st.error("No stocks available in the database.")
    st.stop()

selected_company = st.selectbox(
    "🔍 Search Stock",
    stocks_df["Company"].dropna().unique(),
    index=0
)

stock_name = stocks_df.loc[
    stocks_df["Company"] == selected_company,
    "Symbol"
].iloc[0]


# ---------------------------------
# TOP AI RECOMMENDATIONS
# ---------------------------------

st.subheader("🔥 Top AI Stock Recommendations")

# Safe limit for demo; increase after optimization
top_stocks = stocks_df["Symbol"].dropna().head(50).tolist()

if not top_stocks:
    st.warning("No valid stock symbols found.")
else:
    # Progress bar
    progress_bar = st.progress(0, text="Generating recommendations...")
    
    try:
        recommendations = get_top_recommendations(
            stock_list=top_stocks, 
            stocks_df=stocks_df,
            _progress_bar = progress_bar
            )
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        recommendations = []
    
    progress_bar.empty()


# ---------------------------------
# RECOMMENDATION TABLE
# ---------------------------------

if recommendations:
    recommendation_df = pd.DataFrame(recommendations)[[
        "stock", "signal", "score", "confidence", "accuracy", "reason"
    ]]

    recommendation_df.columns = [
        "Stock", "Signal", "Hybrid Score", 
        "Confidence %", "Model Accuracy %", "Reason"
    ]

    # Emoji signals
    signal_emojis = {
        "BUY": "📈 BUY",
        "SELL": "📉 SELL",
        "HOLD": "⏸️ HOLD"
    }
    recommendation_df["Signal"] = recommendation_df["Signal"].map(signal_emojis).fillna("⏸️ HOLD")

    # Sort and format
    recommendation_df = (
        recommendation_df
        .sort_values(by="Hybrid Score", ascending=False)
        .reset_index(drop=True)
    )
    recommendation_df.index += 1  # Add ranking

    st.dataframe(
        recommendation_df.head(10),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No recommendations generated yet. Please wait or try again later.")


# ---------------------------------
# MAIN PREDICTION
# ---------------------------------

if st.button("🚀 Predict", type="primary"):
    with st.spinner("Fetching and analyzing data..."):
        try:
            data = load_data(stock_name)
        except Exception as e:
            st.error(f"Failed to load data for {stock_name}: {e}")
            st.stop()

    # Validate data
    if data.empty:
        st.error("❌ Invalid stock symbol or no data found.")
        st.stop()

    # Prepare ML data
    try:
        data, X, y, X_train, X_test, y_train, y_test = prepare_data(data)
    except Exception as e:
        st.error(f"Data preparation failed: {e}")
        st.stop()

    # Train model
    try:
        model, acc, name = train_model(X_train, X_test, y_train, y_test)
    except Exception as e:
        st.error(f"Model training failed: {e}")
        st.stop()

    # Latest prediction
    latest = X.tail(1)
    pred = model.predict(latest)

    # Confidence
    try:
        prob = model.predict_proba(latest)
        confidence = round(max(prob[0]) * 100, 2)
    except AttributeError:
        confidence = "N/A (model doesn't support probability)"
        st.warning("⚠️ Model doesn't support probability estimates.")

    # Backtest
    try:
        data = run_backtest(data, model, X)
    except Exception as e:
        st.warning(f"Backtest skipped due to error: {e}")

    # Layout
    col1, col2 = st.columns(2)

    # LEFT SIDE
    with col1:
        try:
            show_candlestick_chart(data)
            show_chart(data)
            show_metrics(data)
        except Exception as e:
            st.error(f"Visualization error: {e}")

        trade_count = (data["Strategy_Return"] != 0).sum()
        st.metric("Completed Trades", trade_count)
        st.write("Latest Close Price:", round(data["Close"].iloc[-1], 2))

    # RIGHT SIDE
    with col2:
        try:
            headlines = fetch_news(selected_company)
        except Exception as e:
            st.error(f"News fetch failed: {e}")
            headlines = []

        if not headlines:
            st.warning("No recent news found.")

        try:
            overall_sentiment, overall_score, headline_results = analyze_overall_sentiment(headlines)
        except Exception as e:
            st.error(f"Sentiment analysis failed: {e}")
            overall_sentiment, overall_score, headline_results = "Unknown", 0.0, []

        try:
            final_signal, final_score, reason = generate_signal(
                pred[0] if isinstance(pred, (list, tuple)) else pred,
                confidence if isinstance(confidence, (int, float)) else 0.0,
                overall_score
            )
        except Exception as e:
            st.error(f"Signal generation failed: {e}")
            final_signal, final_score, reason = "HOLD", 0.0, "Error in signal generation"

        try:
            show_prediction(
                pred, confidence, acc, name,
                final_signal, final_score, reason
            )
        except Exception as e:
            st.error(f"Prediction display failed: {e}")

        st.subheader("Overall Market Mood")
        st.metric("News Sentiment", overall_sentiment)
        st.metric("Sentiment Score", round(overall_score, 2))

        st.subheader("Latest News Sentiment")
        for result in headline_results:
            st.write("📰", result.get("headline", "N/A"))
            st.write("Sentiment:", result.get("sentiment", "N/A"))
            st.write("Score:", round(result.get("score", 0), 2))
            st.divider()