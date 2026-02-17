import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

def load_tsla():
    # 1. Load the data
    df = pd.read_csv("data/processed/tsla.csv", index_col=0, parse_dates=True)
    
    # 2. Set the frequency to Business Days ('B') 
    # This tells Python to expect data only on Mon-Fri
    df = df.asfreq('B')
    
    # 3. Fill missing values (like holidays) using the last known price
    df = df.ffill()
    
    return df["Close"]
def train_test_split(series, split_date="2024-12-31"):
    train = series[series.index <= split_date]
    test  = series[series.index > split_date]
    return train, test


# -----------------------------
# ARIMA MODEL
# -----------------------------
def run_arima(train, test):

    model_auto = auto_arima(
        train,
        seasonal=False,
        trace=False,
        suppress_warnings=True
    )

    order = model_auto.order
    model = ARIMA(train, order=order)
    fitted = model.fit()

    forecast = fitted.forecast(len(test))

    return forecast, order


def evaluate(y_true, y_pred):

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true.values - y_pred.values) / y_true.values)) * 100

    return mae, rmse, mape

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def prepare_lstm(series, window=60):

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1,1))

    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i])
        y.append(scaled[i])

    return np.array(X), np.array(y), scaler


def run_lstm(series):

    X, y, scaler = prepare_lstm(series)

    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(X.shape[1],1)),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    preds = model.predict(X_test)
    preds = scaler.inverse_transform(preds)

    y_test = scaler.inverse_transform(y_test)

    return y_test.flatten(), preds.flatten()
