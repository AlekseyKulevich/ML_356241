import joblib
import numpy as np
import pandas as pd
import tensorflow as keras
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import requests
import json
import time
from datetime import datetime, timedelta

def fetch_binance_klines(symbol: str, interval: str, start_time: int, end_time: int) -> pd.DataFrame:
    """
    Получает исторические данные (свечи) с Binance API и возвращает их в виде DataFrame.

    :param symbol: Торговая пара (например, 'BTCUSDT')
    :param interval: Таймфрейм свечей (например, '1h', '1d')
    :param start_time: Начальное время (в миллисекундах)
    :param end_time: Конечное время (в миллисекундах)
    :return: DataFrame с историческими данными
    """
    url = f"https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": 1000
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Проверяем, не произошла ли ошибка
        data = response.json()
        print(len(data))
        if not data:
            print("Данные не получены. Возможно, некорректные параметры.")
            return pd.DataFrame()

        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume', 'ignore'
        ]
        df = pd.DataFrame(data, columns=columns)

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        df.set_index('timestamp', inplace=True)

        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        print('ffff',df.shape)
        print("Данные успешно загружены.")
        return df

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе данных: {e}")
        return pd.DataFrame()


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Нормализует столбец 'close' с помощью MinMaxScaler.

    :param df: DataFrame с историческими данными, содержащий столбец 'close'
    :return: Кортеж (нормализованный DataFrame, объект MinMaxScaler)
    """
    if 'close' not in df:
        raise ValueError("DataFrame должен содержать столбец 'close' для нормализации.")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['close']])

    print("Данные успешно нормализованы.")
    return scaled_data, scaler


def create_sequences(data, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Преобразует временной ряд в обучающие последовательности для модели LSTM.

    :param data: Одномерный массив с временными рядами.
    :param seq_length: Длина последовательности (количество прошлых значений для предсказания будущего).
    :return: Кортеж (массив признаков X, массив целевых значений Y).
    """
    if len(data) <= seq_length:
        raise ValueError("Длина данных должна быть больше seq_length.")

    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])

    print(f"Создано {len(x)} последовательностей (seq_length={seq_length}).")

    return np.array(x), np.array(y)


def train_lstm_model(symbol: str, interval: str, start_time: int, end_time: int,
                     seq_length: int, lookahead: int):
    """
    Загружает данные, предобрабатывает их, обучает LSTM-модель и сохраняет её вместе с масштабировщиком.

    :param symbol: Торговая пара (например, 'BTCUSDT')
    :param interval: Таймфрейм свечей (например, '1h')
    :param start_time: Начальное время (в миллисекундах Unix)
    :param end_time: Конечное время (в миллисекундах Unix)
    :param seq_length: Длина последовательности для LSTM
    :param lookahead: Количество шагов вперёд для предсказания
    :param epochs: Количество эпох обучения (по умолчанию 10)
    :param batch_size: Размер батча (по умолчанию 32)
    :return: tuple (обученная модель, scaler, mse, предсказания)
    """

    data = fetch_binance_klines(symbol, interval, start_time, end_time)
    print(data)
    scaled_data, scaler = preprocess_data(data)

    x, y = create_sequences(scaled_data, seq_length + lookahead)

    train_size = int(len(x) * 0.8)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(seq_length, 1)),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse')

    model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

    mse = model.evaluate(x_test, y_test)
    print(f"Mean Squared Error: {mse}")

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    model.save('cryptocurrency_lstm_model.keras')
    joblib.dump(scaler, 'cryptocurrency_scaler.pkl')

    print("Обучение завершено. Модель и scaler сохранены.")


def date_to_milliseconds(date_str: str) -> int:
    """
    Преобразует строку даты (в формате 'YYYY-MM-DD') в миллисекунды для Binance API.

    :param date_str: Дата в формате 'YYYY-MM-DD'
    :return: Количество миллисекунд с начала эпохи (1970-01-01)
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp() * 1000)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    symbol = 'BTCUSDT'  # Symbol for Bitcoin-USDT pair
    interval = '1h'  # Interval for historical data (1 hour)

    start_date = "2025-04-05"
    end_date = "2025-05-05"

    start_time = date_to_milliseconds(start_date)
    end_time = date_to_milliseconds(end_date)

    seq_length = 10  # Sequence length for LSTM
    lookahead = 1  # Lookahead period for predicting the future

    train_lstm_model(symbol, interval, start_time, end_time, seq_length, lookahead)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
