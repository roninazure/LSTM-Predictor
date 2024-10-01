import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pandas_market_calendars as mcal
from textblob import TextBlob
import requests
import nltk

#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('brown')

# Corrected function to fetch news sentiment data
def fetch_news_sentiment(NVDA, HV9ULK5AAZ2850SA):
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "apikey": api_key
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to fetch news data")
        return None

tf.random.set_seed(1)

# Assume this is the correct path to your dataset
file_path = '/Users/scottsteele/SMM/NVDA_stock_prices.csv'  # Update with your actual file path
df = pd.read_csv(file_path)

# Preprocessing steps remain unchanged
close_prices = df['Close'].values.reshape(-1, 1)
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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input

# Placeholder for sentiment_scores initialization
sentiment_scores = []

# Assuming 'news_data' contains the fetched news and 'news' in news_data contains the articles
if news_data and 'news' in news_data:
# Example of processing the news data
   for article in news_data['news']:
        # Perform sentiment analysis on each article headline
        headline = article['headline']
        analysis = TextBlob(headline)
        sentiment_score = analysis.sentiment.polarity
        sentiment_scores.append(sentiment_score)

    # Now, sentiment_scores contains the sentiment polarity for each headline

# Ensure sentiment_scores is a numpy array for the next steps
sentiment_scores = np.array(sentiment_scores)

# This assumes sentiment_scores is a 1D numpy array; reshape it as needed
scaled_data_with_sentiment = np.hstack((scaled_data, sentiment_scores.reshape(-1, 1)))

# Now you'll use scaled_data_with_sentiment to create sequences
X, y = create_sequences(scaled_data_with_sentiment, sequence_length)

# Make sure to adjust the input shape in your LSTM model to accommodate the additional sentiment feature
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X.shape[2])))  # Adjusted for an additional feature
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(1))

# Define the LSTM model
model = Sequential()
model.add(Input(shape=(X_train.shape[1], [2])))  # Define input shape explicitly with an Input layer
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(1))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Ensure `news_data` is initialized
news_data = None
# Fetch and preprocess sentiment data (using your API key and symbol)
api_key = "HV9ULK5AAZ2850SA"  # Replace with your actual API key
symbol = "NVDA"
news_data = fetch_news_sentiment(symbol, api_key)

# Proceed with checking and processing `news_data`
if news_data and 'news' in news_data:

# Example sentiment analysis (assuming `news_data` contains the necessary information)
if news_data and 'news' in news_data:
    sentiment_scores = [TextBlob(article['headline']).sentiment.polarity for article in news_data['news']]
    average_sentiment_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    # Adjust data preprocessing to include sentiment scores as needed

model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Adjusted function to find the next 21 valid trading days
def find_next_trading_days(given_date, num_days=21):
    nyse = mcal.get_calendar('NYSE')
    given_date_tz_naive = pd.Timestamp(given_date).tz_localize(None)
    end_date = given_date_tz_naive + pd.Timedelta(days=45)  # Increased buffer to ensure finding at least 21 days
    valid_days = nyse.valid_days(start_date=given_date_tz_naive, end_date=end_date)
    valid_days_naive = [day.tz_localize(None) for day in valid_days]
    next_days = [day for day in valid_days_naive if day > given_date_tz_naive][:num_days]
    return next_days

# Assuming you want to predict the price after a specific date
start_date = '2024-03-28'
next_trading_days = find_next_trading_days(start_date, 21)

predicted_prices = []
current_sequence = scaled_data[-sequence_length:]

# Predict the next 21 days
for _ in range(21):
    current_sequence_reshaped = np.reshape(current_sequence, (1, sequence_length, 1))
    next_price_scaled = model.predict(current_sequence_reshaped)
    next_price = scaler.inverse_transform(next_price_scaled)
    predicted_prices.append(next_price[0][0])
    current_sequence = np.append(current_sequence, next_price_scaled)[-sequence_length:].reshape(-1, 1)

# Print the predicted prices for the next 21 trading days
for i, price in enumerate(predicted_prices):
    print(f"Predicted price for {next_trading_days[i].strftime('%Y-%m-%d')}: {price}")
