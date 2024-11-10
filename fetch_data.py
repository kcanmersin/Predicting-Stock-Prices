# fetch_data.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def fetch_stock_data(ticker, days_back=500):
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    stock_data.reset_index(inplace=True)
    stock_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    output_file = f"{ticker}_data.csv"

    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Existing file {output_file} deleted.")

    stock_data.to_csv(output_file, index=False)
    print(f"Data for {ticker} from {start_date} to {end_date} saved to {output_file}")

if __name__ == "__main__":
    ticker = "AAPL"
    fetch_stock_data(ticker)
