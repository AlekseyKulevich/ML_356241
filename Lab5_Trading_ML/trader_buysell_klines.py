import joblib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras.models
import numpy as np
import pandas as pd
import requests
import json
import time
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)

scaler = joblib.load('cryptocurrency_scaler.pkl')
model = keras.models.load_model('cryptocurrency_lstm_model.keras')


def get_realtime_data(symbol: str, interval: str = '1h') -> float:
    """
    Получает цену закрытия последней свечи для заданного символа и интервала с Binance API.

    :param symbol: Торговая пара (например, 'BTCUSDT').
    :param interval: Интервал свечи (например, '1m', '5m', '1h').
    :return: Цена закрытия последней свечи.
    """
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=1"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        close_price = float(data[0][4])  # Индекс 4 — это close
        return close_price
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при получении данных с Binance: {e}")
        return -1.0



def make_buy_decision(
        model,
        current_data: list[float],
        scaler: MinMaxScaler,
        buy_threshold: float,
        sell_threshold: float,
        last_predicted_price: float
) -> tuple[float, float]:
    """
    Принимает решение о покупке на основе предсказаний модели.

    :param model: Обученная LSTM модель для предсказания цен.
    :param current_data: Последние доступные цены (список).
    :param scaler: Scaler, использованный при обучении модели.
    :param threshold: Пороговое значение изменения предсказания в % для покупки.
    :param last_predicted_price: Последнее предсказанное значение цены.

    :return: Кортеж (True/False — покупать или нет, предсказанная цена).
    """
    # Масштабируем входные данные
    scaled_data = scaler.transform(pd.DataFrame(current_data))

    # Делаем предсказание цены
    prediction = model.predict(np.array([scaled_data]), verbose=0)
    predicted_price = scaler.inverse_transform(prediction)[0][0]

    # Рассчитываем изменение предсказания цены
    if last_predicted_price == 0:
        percentage_change = 0
    else:
        percentage_change = ((predicted_price - last_predicted_price) / last_predicted_price) * 100

    # Логируем результат
    print(f"\nПрошлое предсказание: {last_predicted_price:.2f}, Текущее предсказание: {predicted_price:.2f}, "
          f"Изменение: {percentage_change}%")
    if percentage_change > buy_threshold:
        decision = 1
    elif percentage_change < sell_threshold:
        decision = -1
    else:
        decision = 0

    return decision, predicted_price


def trading():
    """
    Функция для автоматического трейдинга на основе прогноза LSTM-модели.
    """

    symbol = 'BTCUSDT'
    buy_threshold = 0.01  # 0.5% разница для покупки
    sell_threshold = -0.01  # 1.5% разница для продажи
    seq_length = 10  # Длина входных данных

    balance = 100000  # Начальный баланс в USDT
    crypto_balance = 0  # Количество купленной крипты
    initial_balance = balance  # Запоминаем стартовый баланс
    previous_data = [get_realtime_data(symbol)]
    last_predicted_price = 0  # Предыдущее предсказание цены

    print(f"Начальная цена {symbol}: {previous_data[-1]}\n")
    print(f"Стартовый баланс: {balance} USDT\n")
    '''current_price = get_realtime_data(symbol)
    crypto_balance = balance / current_price  # Покупаем на весь баланс
    balance = balance - crypto_balance * current_price
    buy_price = current_price
    previous_buy_price = current_price
    previous_sell_price = 0
    print(f"Куплено {crypto_balance:.6f} BTC по цене {buy_price:.2f} USDT")'''

    flag = 0
    while True:
        try:
            current_price = get_realtime_data(symbol)
            if current_price == -1:
                time.sleep(5)
                continue

            previous_data.append(current_price)

            if len(previous_data) > seq_length:

                current_data = previous_data[-seq_length:]
                buy_decision, predicted_price = make_buy_decision(
                    model, current_data, scaler, buy_threshold, sell_threshold, last_predicted_price
                )

                last_predicted_price = predicted_price  # Обновляем прошлое предсказание

                print("Время:", time.strftime('%Y-%m-%d %H:%M:%S'))
                print(f"Текущая цена: {current_price}")
                print(f"Прогнозируемая цена: {predicted_price}")

                # Покупаем, если есть решение на покупку и у нас есть USDT
                if buy_decision == 1 and balance > 0 and flag == 1:
                    crypto_balance = balance / current_price  # Покупаем на весь баланс
                    balance = 0
                    buy_price = current_price
                    print(f"Куплено {crypto_balance:.6f} BTC по цене {buy_price:.2f} USDT")

                # Продаём, если цена выросла на profit_threshold % и у нас есть крипта
                if buy_decision == -1 and crypto_balance > 0 and flag == 1:
                    balance = crypto_balance * current_price  # Продаем всю крипту
                    print(f"Продано {crypto_balance:.6f} BTC по цене {current_price:.2f} USDT%")
                    crypto_balance = 0
                flag = 1

                time.sleep(3600)  # Ждем час перед следующей проверкой

        except KeyboardInterrupt:
            # Итоговая прибыль
            current_price = get_realtime_data(symbol)
            final_balance = balance + (crypto_balance * current_price)
            profit = final_balance - initial_balance
            print("\nТорговля остановлена вручную.")
            print(f"Конечный баланс: {final_balance:.2f} USDT")
            print(f"Общая прибыль: {profit:.2f} USDT ({(profit / initial_balance) * 100:.2f}%)\n")
            break
        except Exception as e:
            print(f"Ошибка: {e}")
            time.sleep(5)  # Повтор через 5 секунд


if __name__ == '__main__':
    trading()
