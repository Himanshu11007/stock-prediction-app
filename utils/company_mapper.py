COMPANY_MAPPING = {
        "RELIANCE.NS" : "Reliance Industries",
        "INFY.NS":"Infosys",
        "TCS.NS":"Tata Consultancy Services",
        "SBIN.NS":"State bank of India",
        "HDFCBANK.NS":"HDFC Bank",
        "ICICIBANK.NS":"ICIC Bank"
}

def get_company_names(stock_symbol):
    return COMPANY_MAPPING.get(stock_symbol,stock_symbol)

def get_stock_symbol(company_name):
    return list(COMPANY_MAPPING.get(company_name))