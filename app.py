from flask import Flask, request, jsonify
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.metrics import mean_squared_error
app = Flask(__name__)

DATA_DIR = 'data'

@app.route('/fetch_data', methods=['GET'])
def fetch_stock_data():
    ticker = request.args.get('ticker', default='AAPL', type=str).upper()
    days_back = request.args.get('days_back', default=500, type=int)

    ticker_dir = os.path.join(DATA_DIR, ticker)
    if not os.path.exists(ticker_dir):
        os.makedirs(ticker_dir)

    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

    stock_data = yf.download(ticker, start=start_date, end=end_date)
    if stock_data.empty:
        return jsonify({"error": f"No data found for ticker {ticker}"}), 400
    stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    stock_data.reset_index(inplace=True)
    stock_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    output_file = os.path.join(ticker_dir, f"{ticker}_data.csv")

    stock_data.to_csv(output_file, index=False)
    return jsonify({"message": f"Data for {ticker} saved to {output_file}"})


@app.route('/add_indicators', methods=['GET'])
def add_technical_indicators():
    ticker = request.args.get('ticker', default='AAPL', type=str).upper()
    input_file_name = request.args.get('input_file', default=f"{ticker}_data.csv", type=str)

    ticker_dir = os.path.join(DATA_DIR, ticker)
    input_file = os.path.join(ticker_dir, input_file_name)

    if not os.path.exists(input_file):
        return jsonify({"error": f"File {input_file} not found"}), 400

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

    df = df.dropna()

    output_file_name = f"{ticker}_data_with_indicators.csv"
    output_file = os.path.join(ticker_dir, output_file_name)
    df.to_csv(output_file, index=False)
    return jsonify({"message": f"Data with technical indicators saved to {output_file}"})


@app.route('/train_lstm_model', methods=['GET'])
def train_lstm_model():
    ticker = request.args.get('ticker', default='AAPL', type=str).upper()
    input_file_name = request.args.get('input_file', default=f"{ticker}_data.csv", type=str)
    epochs = request.args.get('epochs', default=50, type=int)
    batch_size = request.args.get('batch_size', default=32, type=int)
    seq_len = request.args.get('seq_len', default=60, type=int)

    ticker_dir = os.path.join(DATA_DIR, ticker)
    input_file = os.path.join(ticker_dir, input_file_name)

    if not os.path.exists(input_file):
        return jsonify({"error": f"File {input_file} not found"}), 400

    df = pd.read_csv(input_file)
    data = df[['Close']].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    scaler_filename = os.path.join(ticker_dir, f"{ticker}_scaler.save")
    joblib.dump(scaler, scaler_filename)

    x_train, y_train = [], []
    for i in range(seq_len, len(scaled_data)):
        x_train.append(scaled_data[i - seq_len:i, 0])
        y_train.append(scaled_data[i, 0])
    if len(x_train) == 0:
        return jsonify({"error": f"Not enough data to train the model with sequence length {seq_len}"}), 400
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stop])

    train_predictions = model.predict(x_train)
    rmse = np.sqrt(mean_squared_error(y_train, train_predictions))

    model_filename = os.path.join(ticker_dir, f"{ticker}_lstm_model.h5")
    model.save(model_filename)

    return jsonify({
        "message": f"LSTM model for {ticker} trained and saved as {model_filename}",
        "rmse": rmse
    })


@app.route('/predict_lstm_model', methods=['GET'])
def predict_lstm_model():
    ticker = request.args.get('ticker', default='AAPL', type=str).upper()
    predict_days = request.args.get('predict_days', default=10, type=int)
    seq_len = request.args.get('seq_len', default=60, type=int)
    input_file_name = request.args.get('input_file', default=f"{ticker}_data.csv", type=str)

    ticker_dir = os.path.join(DATA_DIR, ticker)
    input_file = os.path.join(ticker_dir, input_file_name)
    model_filename = os.path.join(ticker_dir, f"{ticker}_lstm_model.h5")
    scaler_filename = os.path.join(ticker_dir, f"{ticker}_scaler.save")

    if not os.path.exists(model_filename) or not os.path.exists(input_file):
        return jsonify({"error": "Model file or input data file not found"}), 400

    model = load_model(model_filename)
    scaler = joblib.load(scaler_filename)  

    df = pd.read_csv(input_file)
    data = df[['Close']].values
    scaled_data = scaler.transform(data)

    if len(scaled_data) < seq_len:
        return jsonify({"error": f"Not enough data to make predictions with sequence length {seq_len}"}), 400

    x_input = scaled_data[-seq_len:]
    x_input = np.reshape(x_input, (1, x_input.shape[0], 1))

    predictions = []
    for _ in range(predict_days):
        pred = model.predict(x_input)
        predictions.append(pred[0][0])
        x_input = np.append(x_input[:, 1:, :], [[[pred[0][0]]]], axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten().tolist()
    return jsonify({"predictions": predictions})


if __name__ == "__main__":
    app.run(debug=True)
