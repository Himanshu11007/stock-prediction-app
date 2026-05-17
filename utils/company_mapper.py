import yfinance as yf

COMPANY_MAPPING = {
        "RELIANCE.NS" : "Reliance Industries",
        "INFY.NS":"Infosys",
        "TCS.NS":"Tata Consultancy Services",
        "SBIN.NS":"State bank of India",
        "HDFCBANK.NS":"HDFC Bank",
        "ICICIBANK.NS":"ICIC Bank",
        "ATHERENERG.NS":"Ather Energy"
}

def get_company_names(stock_symbol):
    try:
        stock = yf.Ticker(stock_symbol)
        info = stock.info

        return info.get(
            "longName",
            stock_symbol.replace(".NS","")
        )
    
    except Exception:
        return stock_symbol.replace(".NS","")

def get_stock_symbol(company_name):
    return list(COMPANY_MAPPING.get(company_name))