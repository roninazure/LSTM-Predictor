import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

# Load the dataset
file_path = '/Users/scottsteele/SMM/TSLA_stock_prices.csv'  # Update with your actual file path
df = pd.read_csv(file_path)

# Assuming 'Close' as the target, using 'Open', 'High', 'Low', 'Close', 'Volume' for features
features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values

# Scale the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

# Define function to create sequences
def create_sequences(data, sequence_length, prediction_length):
    X, y = [], []
    for i in range(len(data) - sequence_length - prediction_length + 1):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length:i+sequence_length+prediction_length, 3])  # Assuming 'Close' is at index 3
    return np.array(X), np.array(y)

sequence_length = 60  # Days to look back for prediction
prediction_length = 5  # Days to predict into the future
X, y = create_sequences(scaled_features, sequence_length, prediction_length)

# Split into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, X.shape[2])),
    Dropout(0.3),
    LSTM(50),
    Dropout(0.3),
    Dense(prediction_length)  # Predicting 5 future values
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Convert holiday strings to Timestamps for accurate comparison
holidays = pd.to_datetime([
    '2024-01-01', '2024-01-15', '2024-02-19', '2024-03-29', 
    '2024-05-27', '2024-06-19', '2024-07-04', '2024-09-02', 
    '2024-11-28', '2024-12-25'
])

# Function to find next business days excluding weekends and holidays
def find_next_business_days(start_date, n_days, holidays):
    business_days = pd.bdate_range(start=start_date, periods=n_days+len(holidays)+30, freq='B')
    valid_days = business_days[~business_days.isin(holidays)][:n_days]
    return valid_days

# Predict for the most recent sequence
last_sequence = scaled_features[-sequence_length:]
last_sequence_reshaped = np.reshape(last_sequence, (1, sequence_length, X.shape[2]))
predicted_prices_scaled = model.predict(last_sequence_reshaped)

# Adjusting the shape for inverse scaling
dummy_features_array = np.zeros((prediction_length, scaled_features.shape[1]))  # Create a dummy array
dummy_features_array[:, 3] = predicted_prices_scaled.flatten()  # Place predictions in the 'Close' column
predicted_prices = scaler.inverse_transform(dummy_features_array)[:, 3]  # Inverse transform and select 'Close'

# Making predictions
last_known_date = '2024-02-29'
start_prediction_date = pd.to_datetime(last_known_date) + pd.Timedelta(days=1)
prediction_dates = find_next_business_days(start_prediction_date, 5, holidays)

# Print predictions for the next 5 business days
for i, price in enumerate(predicted_prices.flatten(), 1):
    print(f"Predicted price for {prediction_dates[i-1].strftime('%Y-%m-%d')}: {price:.2f}")

