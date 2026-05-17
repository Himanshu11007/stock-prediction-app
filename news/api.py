import requests
from utils.company_mapper import get_company_names
import streamlit as st

API_KEY = st.secrets["API_KEY"]

def fetch_news(company_name):

    stock_name = get_company_names(company_name)
    """Fetch latest news headlines"""
    url=(
        f"https://newsapi.org/v2/everything?"
        f"q={stock_name}&"
        f"language=en&"
        f"sortBy=publishedAt&"
        f"apiKey={API_KEY}"
    )

    response = requests.get(url)
    data=response.json()
    articles = data.get("articles",[])
    headlines=[]
    for article in articles[:5]:
        headlines.append(article["title"])
    return headlines
