import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pandas_market_calendars as mcal

tf.random.set_seed(1)

# Prompt the user to enter a ticker symbol
ticker_symbol = input("Please enter the ticker symbol for the stock you want to predict: ").upper()

# Construct the file path dynamically based on the provided ticker symbol
file_path = f'/Users/scottsteele/SMM/{ticker_symbol}_stock_prices.csv'
print(f"Loading data for {ticker_symbol} from {file_path}")

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File not found: {file_path}. Please check the ticker symbol and try again.")
    exit()

df['Date'] = pd.to_datetime(df['Date'])

# Assuming NYSE calendar
nyse = mcal.get_calendar('NYSE')
start_date = df['Date'].min().strftime('%Y-%m-%d')
end_date = (df['Date'].max() + pd.Timedelta(days=365)).strftime('%Y-%m-%d')

trading_days = nyse.valid_days(start_date=start_date, end_date=end_date)
trading_days_df = pd.DataFrame({'valid_days': pd.to_datetime(trading_days.date)})
df = df[df['Date'].isin(trading_days_df['valid_days'])]

close_prices = df['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

def create_sequences(data, sequence_length, prediction_length=5):
    xs, ys = [], []
    for i in range(len(data) - sequence_length - prediction_length + 1):
        x = data[i:(i + sequence_length)]
        y = data[(i + sequence_length):(i + sequence_length + prediction_length), 0]
        xs.append(x)
        ys.append(y[-1])  # Targeting the last day in the prediction sequence for the y value
    return np.array(xs), np.array(ys)

sequence_length = 60
X, y = create_sequences(scaled_data, sequence_length)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

train_size = int(len(X) * 0.8)
X_train, X_test = X[0:train_size], X[train_size:]
y_train, y_test = y[0:train_size], y[train_size:]

model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(sequence_length, 1)))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

last_sequence = scaled_data[-sequence_length:]
last_sequence_reshaped = np.reshape(last_sequence, (1, sequence_length, 1))
predicted_price_scaled = model.predict(last_sequence_reshaped)
predicted_price = scaler.inverse_transform(predicted_price_scaled)

last_known_date = df['Date'].iloc[-1]
next_trading_day = trading_days_df[trading_days_df['valid_days'] > last_known_date]['valid_days'].min()

print(f"Direct 5-day predicted price from {next_trading_day.strftime('%Y-%m-%d')}: {predicted_price[0][0]}")

