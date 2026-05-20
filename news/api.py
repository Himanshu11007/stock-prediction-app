import requests
from utils.company_mapper import get_company_names
import streamlit as st

def fetch_news(company_name):
    """Fetch latest news headlines"""
    api_key = st.secrets["API_KEY"]
    stock_name = get_company_names(company_name)
    url=(
        f"https://newsapi.org/v2/everything?"
        f"q={stock_name}&"
        f"language=en&"
        f"sortBy=publishedAt&"
        f"apiKey={api_key}"
    )

    response = requests.get(url)
    response.raise_for_status()
    data=response.json()
    articles = data.get("articles",[])
    headlines=[]
    for article in articles[:5]:
        headlines.append(article["title"])
    return headlines
