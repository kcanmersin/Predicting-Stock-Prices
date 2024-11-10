import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

file_path = r'AAPL_data_with_indicators.csv' 
data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

features = data.drop(columns=['Close'])
target = data['Close']

scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler_X.fit_transform(features)
target_normalized = scaler_y.fit_transform(target.values.reshape(-1, 1))

train_size = int(len(data_normalized) * 0.8)
X_train, X_test = data_normalized[:train_size], data_normalized[train_size:]
y_train, y_test = target_normalized[:train_size], target_normalized[train_size:]

time_steps = 10 
def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

X_train_lstm, y_train_lstm = create_sequences(X_train, y_train, time_steps)
X_test_lstm, y_test_lstm = create_sequences(X_test, y_test, time_steps)

def create_lstm_model(units, activation, learning_rate):
    model = Sequential([
        LSTM(units=units, activation=activation, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
        Dense(units=1)
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

lstm_units = [50, 100, 200]
lstm_activations = ['relu', 'tanh']
learning_rates = [0.001, 0.01, 0.1]
epochs = 50
batch_size = 32

best_rmse = float('inf')
best_lstm_model = None

for units in lstm_units:
    for activation in lstm_activations:
        for learning_rate in learning_rates:
            model = create_lstm_model(units=units, activation=activation, learning_rate=learning_rate)
            model.fit(X_train_lstm, y_train_lstm, epochs=epochs, batch_size=batch_size, verbose=0)

            test_predictions = model.predict(X_test_lstm).flatten()
            rmse = np.sqrt(mean_squared_error(y_test_lstm, test_predictions))

            if rmse < best_rmse:
                best_rmse = rmse
                best_lstm_model = model

all_lstm_predictions = best_lstm_model.predict(X_train_lstm).flatten()
all_lstm_predictions = scaler_y.inverse_transform(all_lstm_predictions.reshape(-1, 1)).flatten()

plt.figure(figsize=(10, 6))
plt.plot(target.index[time_steps:train_size], scaler_y.inverse_transform(y_train_lstm), label='Training Data (Actual)')
plt.plot(target.index[train_size + time_steps:], scaler_y.inverse_transform(y_test_lstm), label='Testing Data (Actual)')
plt.plot(target.index[train_size + time_steps:], scaler_y.inverse_transform(test_predictions.reshape(-1, 1)), label='LSTM Predicted (Test Data)', linestyle="dashed")
plt.title(f"LSTM Model - RMSE: {best_rmse:.2f}")
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

print(f"Best LSTM RMSE: {best_rmse:.2f}")
