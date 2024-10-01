import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import ta  # Ensure TA-Lib is installed for technical analysis features
import pandas_market_calendars as mcal  # For handling market calendars

from sklearn.preprocessing import MinMaxScaler

tf.random.set_seed(1)  # For reproducibility

# Load your dataset
file_path = '/Users/scottsteele/SMM/SMCI_stock_prices.csv'  # Update this path
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])  # Ensure 'Date' is in datetime format

# Calculate technical indicators
df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

# Drop rows with NaN values
df.dropna(inplace=True)

# Prepare the features and target
features = df[['Close', 'SMA_10', 'SMA_50', 'RSI']].values
target = df['Close'].values

# Scale the features and target
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler_features.fit_transform(features)
scaler_target = MinMaxScaler(feature_range=(0, 1))
scaled_target = scaler_target.fit_transform(target.reshape(-1, 1))

# Adjust these parameters according to your dataset
sequence_length = 30
prediction_length = 1

# Define a function to create sequences
def create_sequences(features, target, sequence_length, prediction_length):
    xs, ys = [], []
    for i in range(len(features) - sequence_length - prediction_length + 1):
        x = features[i:(i + sequence_length)]
        y = target[i + sequence_length + prediction_length - 1]  # Targeting 5 days ahead
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X, y = create_sequences(scaled_features, scaled_target.flatten(), sequence_length, prediction_length)

# Check if X and y have been populated
if len(X) == 0 or len(y) == 0:
    raise ValueError("X or y is empty. Adjust sequence_length and prediction_length.")

# Load the NYSE calendar
nyse = mcal.get_calendar('NYSE')

# Assuming the last known date is in your dataframe
last_known_date = df['Date'].iloc[-1]

# Find the next 5 trading days from the last known date
future_trading_days = nyse.valid_days(start_date=last_known_date, end_date=last_known_date + pd.DateOffset(days=30))
next_5th_trading_day = future_trading_days[4]  # The 5th trading day after the last known date

print(f"Model will predict the stock price for: {next_5th_trading_day.date()}")

# Define the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, 4)),  # Ensure this matches your feature count
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

# Compile the model using the legacy Adam optimizer for better performance on M1/M2 Macs
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=100, batch_size=32, verbose=1)

# Assuming you want to predict using the most recent sequence
last_sequence = scaled_features[-sequence_length:]
last_sequence_reshaped = np.reshape(last_sequence, (1, sequence_length, 4))  # Reshape for prediction
predicted_price_scaled = model.predict(last_sequence_reshaped)
predicted_price = scaler_target.inverse_transform(predicted_price_scaled)

print(f"Predicted price for {next_5th_trading_day.date()}: {predicted_price[0][0]}")

