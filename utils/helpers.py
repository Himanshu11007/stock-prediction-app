import streamlit as st
import plotly.graph_objects as go
from features.engineer import(
    create_features
)
# -------------------------------
# HELPER FUNCTIONS
# -------------------------------

def show_candlestick_chart(data):
    """Display professional candlestick chart using plotly"""

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="Price"
            )
        ]
    )

    fig.update_layout(
        title="Stock Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        height=500,
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig,use_container_width=True)

    
def prepare_data(data):
    """
    Create features and split data into train/test
    """
    # Create ML features
    data = create_features(data)

    # Select input features
    X = data[["Close", "Volume", "Price_Change", "MA_5", "MA_10", "RSI","Momentum","Volatility","Volume_Change","Volume_MA","MA_Diff"]]

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

    # Predict for all data
    data["Prediction"] = model.predict(X)

    # Get probability for confidence
    proba = model.predict_proba(X)

    # Max probability (confidence)
    data["Confidence"] = proba.max(axis=1)

    # Shift prediction to avoid future leakage
    data["Prediction"] = data["Prediction"].shift(1)

    # Apply confidence filter (only strong signals)
    data["Prediction"] = (
        (data["Prediction"] == 1) &
        (data["Confidence"] > 0.6) &
        (data["MA_5"] > data["MA_10"] * 1.01)
        ).astype(int)

    # Market return (daily % change)
    data["Market_Return"] = data["Close"].pct_change()

    # Strategy strategy return column
    data["Strategy_Return"] = 0.0

    #Trading Settings
    stop_loss = -0.02
    take_profit = 0.05

    #Track whether we are insidea trade
    in_trade = False
    entry_price = 0

    #Loop through in rows

    for i in range(1,len(data)):
        current_price = data["Close"].iloc[i]

        #Enter trade
        if (not in_trade and data["Prediction"].iloc[i] == 1 and 40 < data ["RSI"].iloc[i] < 70):
            in_trade = True
            entry_price = current_price

        #Manage Trade
        elif in_trade:
            #Calculate profit/loss %
            trade_return = (current_price - entry_price) / entry_price

            #EXIT CONDITIONS
            if trade_return <= stop_loss or trade_return >=take_profit:
                data.loc[data.index[i],"Strategy_Return"] = trade_return
                in_trade=False
        

    # Cumulative growth
    data["Total_Return"] = (1 + data["Strategy_Return"].fillna(0)).cumprod()
    # Market Return
    data["Market_Return"] = data["Close"].pct_change()
    
    #Buy & hold growth
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


def show_prediction(pred, confidence, acc, name,final_signal,final_score):
    """
    Show BUY/SELL signal UI
    """
    st.subheader("📌 Trading Signal")

    #Hold Signal
    if (final_signal == "HOLD" or confidence < 65 or acc < 0.70 or abs(final_score) < 0.25):
        hold_reason = ""
        if confidence < 65:
            hold_reason = "Model confidence is low"
        elif acc < 0.70:
            hold_reason ="Model Accuracy is weak."
        elif abs(final_score) < 0.25:
            hold_reason = "Hybrid score is too weak."
        elif final_signal == "HOLD":
            hold_reason="Marker directin is uncertauin"
        st.markdown(f"""
        <div style="background-color:#fff3cd;padding:20px;border-radius:10px;border-left:6px solid #ffc107;box-shadow:0 2px 4px rgba(0,0,0,0.1)">
        <h2 style="Margin:0;">Hold Signal</h2>
                <p style ="font-size:18px;">
                    <b>Confidence:</b> {confidence}%
                </p>
                 <p style ="font-size:18px;">
                    <b>Hybrid Score:</b> {round(final_score,2)}%
                </p>
                <p>
                    {hold_reason}
                </p>
        </div>
        """, unsafe_allow_html=True)

    #BUY Signal
    elif pred[0] == 1 and final_signal == "BUY":
        if final_score > 0.75 and confidence > 84:
            signal_title = "🚀Strong Buy Signal"
        else:
            signal_title = "📈 BUY Signal"

        st.markdown(f"""
        <div style="background-color:#d4edda;padding:20px;border-radius:10px;border-left:6px solid green;box-shadow:0 2px 4px rgba(0,0,0,0.1)">
        <h2>signal_title</h2>
                <p style ="font-size:18px;">
                    <b>Confidence:</b> {confidence}%
                </p>
                 <p style ="font-size:18px;">
                    <b>Hybrid Score:</b> {round(final_score,2)}%
                </p>
        <p>Models expects bullish movement.</p>
        </div>
        """, unsafe_allow_html=True)

    #Sell Signal   
    elif (pred[0] == 0 and final_signal == "SELL"):
        if final_score < -0.75 and confidence > 85:
            signal_title ="🔥Strong Sell Signal"
        else:
            signal_title ="📉 SELL Signal"
        st.markdown(f"""
        <div style="background-color:#f8d7da;padding:20px;border-radius:10px;border-left:6px solid red;box-shadow:0 2px 4px rgba(0,0,0,0.1)">
        <h2>signal_title</h2>
                <p style ="font-size:18px;">
                    <b>Confidence:</b> {confidence}%
                </p>
                 <p style ="font-size:18px;">
                    <b>Hybrid Score:</b> {round(final_score,2)}%
                </p>
                <p>Model expects bearish movement</p>
        </div>
        """, unsafe_allow_html=True)
    else:
         st.markdown(f"""
        <div style="background-color:#e2e3e5;padding:20px;border-radius:10px;border-left:6px solid #ffc107;box-shadow:0 2px 4px rgba(0,0,0,0.1)">
        <h2 style="Margin:0;">Hold / Uncertain Signal</h2>
                <p style ="font-size:18px;">
                    <b>Confidence:</b> {confidence}%
                </p>
                 <p style ="font-size:18px;">
                    <b>Hybrid Score:</b> {round(final_score,2)}%
                </p>
                <p>
                    Model confidence is low.
                    Waiting may be safer than entering a trade
                </p>
        </div>
        """, unsafe_allow_html=True)

    # Show model info
    st.progress(int(confidence))
    st.metric("Model Accuracy", f"{round(acc * 100, 2)}%")
    st.caption(f"Model Used: {name}")
