import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Stock Predictor",
    layout="wide"
)

st.title("Stock Prediction App")
st.markdown("Predict next-day stock movement using machine learning")

# Input from user
stock_name = st.selectbox(
    "Select Stock",["RELIANCE.NS","TCS.NS","INFY.NS"]
    )

col1,col2 = st.columns(2)

if st.button("Predict"):
    with st.spinner("Fetching and analyzing data.."):

    # Download stock data
        data = yf.download(stock_name, start="2022-01-01")

    if data.empty:
        st.error("Invalid stock name or no data found")
    else:
        # Flatten MultiIndex columns (if present)
        data.columns = data.columns.get_level_values(0)

        # Keep only required columns
        data = data[["Close", "Volume"]]

        # Create target column
        # 1 = next day price goes up
        # 0 = next day price does not go up
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

        # Remove empty rows (drop NaNs)
        data = data.dropna()

        # Debug: Print data after feature engineering
        print("\nData after creating target columns:")
        print(data.head())

        # Define features and target
        X = data[["Close", "Volume", "Price_Change", "MA_5", "MA_10", "RSI"]]
        y = data["Up"]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train Logistic Regression model
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_train, y_train)
        lr_acc = lr_model.score(X_test, y_test)

        # Train Random Forest model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_acc = rf_model.score(X_test, y_test)

        print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
        print(f"Random Forest Accuracy: {rf_acc:.4f}")

        # Determine best model
        best_model = rf_model if rf_acc > lr_acc else lr_model
        best_acc = max(lr_acc,rf_acc)
        
        #predict latest
        latest_data = X.tail(1)
        latest_prediction = best_model.predict(latest_data)

        #confidence
        prob = best_model.predict_proba(latest_data)
        confidence = round(max(prob[0]) * 100, 2)

        with col1:
            st.subheader("Price chart")
            chart_data = data[["Close","MA_5","MA_10"]]
            st.line_chart(chart_data)
            st.write("Latest Close Price:",round(data["Close"].iloc[-1],2))
        with col2:
            st.subheader("Treading Signal")
            if latest_prediction[0] == 1:
               st.markdown(f"""
               <div style="background-color: #d4edda;padding: 20px;border-radius: 10px;border-left: 6px solid #28a745;box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
               <h2 style="color: #155724; margin: 0;">📈 BUY Signal</h2>
               <p style="color: #155724; font-size: 1.1em; margin: 8px 0 0 0;"><strong>Confidence:</strong> {confidence}%</p>
              </div>
               """,unsafe_allow_html=True)
            else:
               st.markdown(f"""
               <div style="background-color: #f8d7da;padding: 20px;border-radius: 10px;border-left: 6px solid #dc3545;box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
               <h2 style="color: #721c24; margin: 0;">📉 SELL Signal</h2>
               <p style="color: #721c24; font-size: 1.1em; margin: 8px 0 0 0;"><strong>Confidence:</strong> {confidence}%</p>
               </div>
               """,unsafe_allow_html=True)

            st.metric("Best Model Accuracy",f"{round(best_acc*100,2)}%")
            