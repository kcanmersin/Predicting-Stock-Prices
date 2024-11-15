from flask import Flask, request, jsonify
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf

app = Flask(__name__)

DATA_DIR = 'data'

class StockModel:
    def __init__(self, ticker, days_back=500):
        self.ticker = ticker.upper()
        self.days_back = days_back
        self.daily_dir = self.get_daily_subdir()
        self.input_file = os.path.join(self.daily_dir, f"{self.ticker}_data.csv")
        self.indicators_file = os.path.join(self.daily_dir, f"{self.ticker}_data_with_indicators.csv")
        self.scaler_filename = os.path.join(self.daily_dir, f"{self.ticker}_scaler.save")
        self.model_filename = os.path.join(self.daily_dir, f"{self.ticker}_lstm_model.keras")

    def get_daily_subdir(self):
        """Generate a daily subdirectory path for the given ticker."""
        today = datetime.now().strftime('%Y-%m-%d')
        subdir = os.path.join(DATA_DIR, self.ticker, today)
        os.makedirs(subdir, exist_ok=True)
        return subdir

    def fetch_data(self):
        """Fetch historical stock data and save to CSV."""
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=self.days_back)).strftime('%Y-%m-%d')
        stock_data = yf.download(self.ticker, start=start_date, end=end_date)
        if stock_data.empty:
            raise ValueError(f"No data found for ticker {self.ticker}")
        stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        stock_data.reset_index(inplace=True)
        stock_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        stock_data.to_csv(self.input_file, index=False)

    def add_indicators(self):
        """Add technical indicators to the stock data."""
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"File {self.input_file} not found")
        df = pd.read_csv(self.input_file, parse_dates=['Date'])
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
        df.dropna(inplace=True)
        df.to_csv(self.indicators_file, index=False)

    def prepare_training_data(self, seq_len):
        """Prepare data for training the LSTM model."""
        if not os.path.exists(self.indicators_file):
            raise FileNotFoundError(f"File {self.indicators_file} not found")
        df = pd.read_csv(self.indicators_file)
        feature_columns = df.columns.difference(['Date', 'Adj Close'])
        data = df[feature_columns].values
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(data)
        joblib.dump(self.scaler, self.scaler_filename)
        x_data, y_data = [], []
        close_index = list(feature_columns).index('Close')
        for i in range(seq_len, len(scaled_data)):
            x_data.append(scaled_data[i - seq_len:i])
            y_data.append(scaled_data[i, close_index])
        if not x_data:
            raise ValueError(f"Not enough data to train the model with sequence length {seq_len}")
        return np.array(x_data), np.array(y_data), feature_columns

    def train_model(self, x_train, y_train, x_val, y_val, seq_len, feature_columns, epochs, batch_size, validation_split, learning_rate, dropout_rate):
        """Train the LSTM model."""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(seq_len, len(feature_columns))),
            Dropout(dropout_rate),
            LSTM(64),
            Dropout(dropout_rate),
            Dense(1)
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        checkpoint = ModelCheckpoint(self.model_filename, monitor='val_loss', save_best_only=True, verbose=1)
        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            callbacks=[early_stop, checkpoint]
        )
        return history

    def delete_old_data(self):
        """Delete previous day's data to save space."""
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        yesterday_dir = os.path.join(DATA_DIR, self.ticker, yesterday)
        if os.path.exists(yesterday_dir):
            for file in os.listdir(yesterday_dir):
                os.remove(os.path.join(yesterday_dir, file))
            os.rmdir(yesterday_dir)

    def predict(self, predict_days, seq_len):
        """Make predictions using the trained LSTM model."""
        if not os.path.exists(self.model_filename) or not os.path.exists(self.indicators_file):
            raise FileNotFoundError("Model file or input data file not found")
        model = load_model(self.model_filename)
        scaler = joblib.load(self.scaler_filename)
        df = pd.read_csv(self.indicators_file)
        feature_columns = df.columns.difference(['Date', 'Adj Close'])
        data = df[feature_columns].values
        scaled_data = scaler.transform(data)
        if len(scaled_data) < seq_len:
            raise ValueError(f"Not enough data to make predictions with sequence length {seq_len}")
        x_input = scaled_data[-seq_len:]
        predictions = []
        close_index = list(feature_columns).index('Close')
        for _ in range(predict_days):
            x_input_reshaped = np.reshape(x_input, (1, seq_len, len(feature_columns)))
            pred = model.predict(x_input_reshaped)
            next_input = x_input[-1].copy()
            next_input[close_index] = pred[0][0]
            x_input = np.vstack((x_input[1:], next_input))
            predictions.append(pred[0][0])
        predictions_array = np.zeros((len(predictions), len(feature_columns)))
        predictions_array[:, close_index] = predictions
        predictions = scaler.inverse_transform(predictions_array)[:, close_index].flatten().tolist()
        return predictions

@app.route('/fetch_data', methods=['GET'])
def fetch_stock_data():
    ticker = request.args.get('ticker', default='AAPL', type=str)
    days_back = request.args.get('days_back', default=500, type=int)
    try:
        stock_model = StockModel(ticker, days_back)
        stock_model.fetch_data()
        return jsonify({"message": f"Data for {ticker.upper()} saved to {stock_model.input_file}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/add_indicators', methods=['GET'])
def add_technical_indicators():
    ticker = request.args.get('ticker', default='AAPL', type=str)
    try:
        stock_model = StockModel(ticker)
        stock_model.add_indicators()
        return jsonify({"message": f"Data with technical indicators saved to {stock_model.indicators_file}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/train_lstm_model', methods=['GET'])
def train_lstm_model():
    ticker = request.args.get('ticker', default='AAPL', type=str)
    epochs = request.args.get('epochs', default=100, type=int)
    batch_size = request.args.get('batch_size', default=32, type=int)
    seq_len = request.args.get('seq_len', default=60, type=int)
    validation_split = request.args.get('validation_split', default=0.1, type=float)
    learning_rate = request.args.get('learning_rate', default=0.001, type=float)
    dropout_rate = request.args.get('dropout_rate', default=0.2, type=float)
    try:
        stock_model = StockModel(ticker)
        x_data, y_data, feature_columns = stock_model.prepare_training_data(seq_len)
        x_train, x_val, y_train, y_val = train_test_split(
            x_data, y_data, test_size=validation_split, shuffle=False
        )
        history = stock_model.train_model(
            x_train, y_train, x_val, y_val, seq_len, feature_columns,
            epochs, batch_size, validation_split, learning_rate, dropout_rate
        )
        val_predictions = load_model(stock_model.model_filename).predict(x_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
        return jsonify({
            "message": f"LSTM model for {ticker.upper()} trained and saved as {stock_model.model_filename}",
            "rmse": rmse,
            "history": {
                "loss": history.history['loss'],
                "val_loss": history.history['val_loss']
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/complete_training', methods=['GET'])
def complete_training():
    ticker = request.args.get('ticker', default='AAPL', type=str)
    days_back = request.args.get('days_back', default=500, type=int)
    epochs = request.args.get('epochs', default=100, type=int)
    batch_size = request.args.get('batch_size', default=32, type=int)
    seq_len = request.args.get('seq_len', default=60, type=int)
    validation_split = request.args.get('validation_split', default=0.1, type=float)
    learning_rate = request.args.get('learning_rate', default=0.001, type=float)
    dropout_rate = request.args.get('dropout_rate', default=0.2, type=float)
    try:
        stock_model = StockModel(ticker, days_back)
        stock_model.fetch_data()
        stock_model.add_indicators()
        x_data, y_data, feature_columns = stock_model.prepare_training_data(seq_len)
        x_train, x_val, y_train, y_val = train_test_split(
            x_data, y_data, test_size=validation_split, shuffle=False
        )
        history = stock_model.train_model(
            x_train, y_train, x_val, y_val, seq_len, feature_columns,
            epochs, batch_size, validation_split, learning_rate, dropout_rate
        )
        stock_model.delete_old_data()
        return jsonify({
            "message": f"Training complete for {ticker.upper()}. Old data deleted.",
            "model_file": stock_model.model_filename,
            "history": {
                "loss": history.history['loss'],
                "val_loss": history.history['val_loss']
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_lstm_model', methods=['GET'])
def predict_lstm_model():
    ticker = request.args.get('ticker', default='AAPL', type=str)
    predict_days = request.args.get('predict_days', default=10, type=int)
    seq_len = request.args.get('seq_len', default=60, type=int)
    try:
        stock_model = StockModel(ticker)
        predictions = stock_model.predict(predict_days, seq_len)
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
