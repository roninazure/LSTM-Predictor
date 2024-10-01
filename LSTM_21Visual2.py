import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pandas_market_calendars as mcal
import matplotlib.pyplot as plt

tf.random.set_seed(1)

# Load the dataset
file_path = '/Users/scottsteele/SMM/TSLA_stock_prices.csv'  # Update with your actual file path
df = pd.read_csv(file_path)

# Select the feature to predict, usually 'Close' for stock prices
close_prices = df['Close'].values.reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

def create_sequences(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data)-sequence_length-1):
        x = data[i:(i+sequence_length), 0]
        y = data[i+sequence_length, 0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

sequence_length = 62
X, y = create_sequences(scaled_data, sequence_length)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

train_size = int(len(X) * 0.8)
X_train, X_test = X[0:train_size,:], X[train_size:len(X),:]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

def find_next_trading_days(given_date, num_days=21):
    nyse = mcal.get_calendar('NYSE')
    given_date_tz_naive = pd.Timestamp(given_date).tz_localize(None)
    end_date = given_date_tz_naive + pd.Timedelta(days=45)
    valid_days = nyse.valid_days(start_date=given_date_tz_naive, end_date=end_date)
    valid_days_naive = [day.tz_localize(None) for day in valid_days]
    next_days = [day for day in valid_days_naive if day > given_date_tz_naive][:num_days]
    return next_days

start_date = '2024-03-28'
next_trading_days = find_next_trading_days(start_date, 21)

predicted_prices = []
current_sequence = scaled_data[-sequence_length:]

for _ in range(21):
    current_sequence_reshaped = np.reshape(current_sequence, (1, sequence_length, 1))
    next_price_scaled = model.predict(current_sequence_reshaped)
    next_price = scaler.inverse_transform(next_price_scaled)
    predicted_prices.append(next_price[0][0])
    current_sequence = np.append(current_sequence, next_price_scaled)[-sequence_length:].reshape(-1, 1)

actual_prices = close_prices[-21:]
predicted_prices_past = model.predict(X[-21:])
predicted_prices_past = scaler.inverse_transform(predicted_prices_past)

dates_past = pd.date_range(end=start_date, periods=21, freq='B').normalize()
dates_future = pd.to_datetime(next_trading_days)

ticker_symbol = 'TSLA'  # Replace 'YourTickerSymbol' with your actual ticker symbol, e.g., 'NVDA'

plt.figure(figsize=(14, 7))
plt.plot(dates_past, actual_prices.flatten(), label='Actual Prices', color='blue')
plt.plot(dates_past, predicted_prices_past.flatten(), label='Predicted Prices (Past)', color='red', linestyle='dashed')
plt.plot(dates_future, predicted_prices, label='Predicted Prices (Future)', color='green', linestyle='dashed')

# Highlight and annotate the most recent actual and predicted prices
most_recent_date = dates_past[-1]
most_recent_actual_price = actual_prices[-1][0]
most_recent_predicted_price = predicted_prices_past[-1][0]
plt.scatter(most_recent_date, most_recent_actual_price, color='gold', label='Most Recent Actual Price', zorder=5)
plt.scatter(most_recent_date, most_recent_predicted_price, color='magenta', label='Most Recent Predicted Price', zorder=5)
plt.annotate(f'{most_recent_date.strftime("%Y-%m-%d")}\n${most_recent_actual_price:.2f}',
             (most_recent_date, most_recent_actual_price), textcoords="offset points", xytext=(-50,10), ha='center', color='gold')
plt.annotate(f'\n${most_recent_predicted_price:.2f}',
             (most_recent_date, most_recent_predicted_price), textcoords="offset points", xytext=(-50,-15), ha='center', color='magenta')

plt.title(f'{ticker_symbol} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

