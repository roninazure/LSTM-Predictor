"""
Stock Price Prediction Script
Copyright (c) Jan 5, 2024, by Scott Steele

This script is designed to predict future stock prices using Long Short-Term Memory (LSTM) neural networks.
It loads historical stock price data, processes and scales the data, and then trains an LSTM model to predict
the stock's closing price for the next valid trading day, taking into account weekends and market holidays.
The script utilizes the pandas, NumPy, Keras, and pandas_market_calendars libraries to prepare data,
define and train the model, and identify the next trading day. It's tailored for use with daily stock price data,
specifically focusing on the 'Close' price as the feature of interest.

Usage:
- Update 'file_path' with the path to your CSV file containing stock price data.
- Ensure the CSV includes a 'Close' column with closing prices.
- The script predicts the price for the next trading day after the date specified in 'start_date'.
- Modify 'start_date' as needed to predict prices for different days.

Note: This script is intended for educational and informational purposes only and does not constitute financial advice.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pandas_market_calendars as mcal

tf.random.set_seed(1)

# Load the dataset
file_path = '/Users/scottsteele/SMM/SMCI_stock_prices.csv'  # Update with your actual file path
df = pd.read_csv(file_path)

# Select the feature to predict, usually 'Close' for stock prices
close_prices = df['Close'].values.reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Function to create sequences for LSTM
def create_sequences(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data)-sequence_length-1):
        x = data[i:(i+sequence_length), 0]
        y = data[i+sequence_length, 0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Create the sequences
sequence_length = 60  # Number of days to look back for prediction
X, y = create_sequences(scaled_data, sequence_length)

# Reshape the data for LSTM layer
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split data into training and test sets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size,:], X[train_size:len(X),:]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Adjusted function to find the next 45 valid trading days
def find_next_trading_days(given_date, num_days=45):
    nyse = mcal.get_calendar('NYSE')
    given_date_tz_naive = pd.Timestamp(given_date).tz_localize(None)
    # Increase the buffer more to ensure finding at least 45 days
    end_date = given_date_tz_naive + pd.Timedelta(days=90)  # Adjusted buffer to account for weekends and holidays
    valid_days = nyse.valid_days(start_date=given_date_tz_naive, end_date=end_date)
    valid_days_naive = [day.tz_localize(None) for day in valid_days]
    next_days = [day for day in valid_days_naive if day > given_date_tz_naive][:num_days]
    return next_days

# Assuming you want to predict the price after a specific date
start_date = '2024-03-08'
next_trading_days = find_next_trading_days(start_date, 45)

predicted_prices = []
current_sequence = scaled_data[-sequence_length:]

# Predict the next 45 days
for _ in range(45):
    current_sequence_reshaped = np.reshape(current_sequence, (1, sequence_length, 1))
    next_price_scaled = model.predict(current_sequence_reshaped)
    next_price = scaler.inverse_transform(next_price_scaled)
    predicted_prices.append(next_price[0][0])
    # Make sure to append the predicted value correctly to maintain the sequence length
    current_sequence = np.append(current_sequence, next_price_scaled)[-sequence_length:].reshape(-1, 1)

# Print the predicted prices for the next 45 trading days
for i, price in enumerate(predicted_prices):
    print(f"Predicted price for {next_trading_days[i].strftime('%Y-%m-%d')}: {price}")

