import yfinance as yf
import streamlit as st

@st.cache_data(ttl=3600)
def load_data(stock_name):
    data = yf.download(stock_name,period="5y")

    data.columns =[col[0]if isinstance(col,tuple) else col for col in data.columns]
    return data
