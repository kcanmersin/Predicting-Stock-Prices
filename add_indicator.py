# add_indicators.py
import pandas as pd
import pandas_ta as ta
import os

def add_technical_indicators(input_file):
    df = pd.read_csv(input_file, parse_dates=['Date'])

    df['SMA_10'] = ta.sma(df['Close'], length=10)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['EMA_10'] = ta.ema(df['Close'], length=10)
    df['EMA_50'] = ta.ema(df['Close'], length=50)

    bbands = ta.bbands(df['Close'], length=20)
    df['Bollinger_Upper'] = bbands['BBU_20_2.0']
    df['Bollinger_Middle'] = bbands['BBM_20_2.0']
    df['Bollinger_Lower'] = bbands['BBL_20_2.0']

    df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['RSI_14'] = ta.rsi(df['Close'], length=14)

    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']

    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['ADX_14'] = adx['ADX_14']
    df['DI+_14'] = adx['DMP_14']
    df['DI-_14'] = adx['DMN_14']

    df['CCI_14'] = ta.cci(df['High'], df['Low'], df['Close'], length=14)

    base_name, ext = os.path.splitext(input_file)
    output_file = f"{base_name}_with_indicators{ext}"

    df = df.dropna(subset=[
        'SMA_10', 'SMA_50', 'EMA_10', 'EMA_50', 
        'Bollinger_Upper', 'ATR_14', 'RSI_14', 
        'MACD', 'MACD_Signal', 'ADX_14', 'DI+_14', 
        'DI-_14', 'CCI_14'
    ])

    df.to_csv(output_file, index=False)
    print(f"Data with technical indicators saved to {output_file}")

if __name__ == "__main__":
    input_file = "AAPL_2023-09-22_2024-11-09_data.csv"
    add_technical_indicators(input_file)
