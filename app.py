import streamlit as st
from data.loader import load_data
from features.engineer import create_features
from  models.trainer import train_model

st.set_page_config(
    page_title="Stock Predictor",
    layout="wide"
)

st.caption("Note : this is an experimental ML Model. Not financial advice")

st.title("Stock Prediction App")
st.markdown("Predict next-day stock movement using machine learning")

# Input from user
stock_name = st.selectbox(
    "Select Stock",["RELIANCE.NS","TCS.NS","INFY.NS"]
    )

col1,col2 = st.columns(2)

if st.button("Predict"):
    with st.spinner("Fetching and analyzing data.."):
        data = load_data(stock_name)
    if data.empty:
        st.error("Invalid stock name or no data found")
    else:
        data = create_features(data)


        # Define features and target
        X = data[["Close", "Volume", "Price_Change", "MA_5", "MA_10", "RSI"]]
        y = data["Up"]

        split = int(len(X)*0.8)
        X_train,X_test = X[:split],X[split:]
        y_train,y_test = y[:split],y[split:]

        model,acc,name = train_model(X_train,y_train,X_test,y_test)
        latest = X.tail(1)
        pred = model.predict(latest)

        prob = model.predict_proba(latest)
        confidence = round(max(prob[0]) * 100 ,2)

        st.write("Model Used:",name)
        st.write("Accuracy:",round(acc * 100,2),"%")

        col1,col2 = st.columns(2)

        with col1:
            st.subheader("Price chart")
            chart_data = data[["Close","MA_5","MA_10"]]
            st.line_chart(chart_data)
            st.write("Latest Close Price:",round(data["Close"].iloc[-1],2))
        with col2:
            st.subheader("Treading Signal")
            if pred[0] == 1:
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

            st.metric("Best Model Accuracy",f"{round(acc*100,2)}%")
            st.caption(f"Model Used:{name}")
            
