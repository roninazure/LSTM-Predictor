import requests
import csv

#Alpha Vantage API key
api_key = 'S4FXZ6TI6EXJ44RB'
symbols = tickers = ['INTC', 'LLY', 'GOOGL', 'NFLX', 'COIN', 'MSFT', 'AMD', 'TSLA', 'AMZN', 'META', 'SPY', 'SMCI', 'NVDA', 'QQQ', 'MSTR', 'MU']  # List of stock symbols

def save_data_to_csv(symbol, data):
    # Extract the 'Time Series (Daily)' portion of the data
    time_series = data.get('Time Series (Daily)', {})
    
    # Define the filename dynamically based on the stock symbol
    file_name = f'{symbol}_stock_prices.csv'
    
    # Open a new CSV file to write to
    with open(file_name, 'w', newline='') as csv_file:
        # Create a CSV writer object
        writer = csv.writer(csv_file)
        
        # Write the header row
        writer.writerow(['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Iterate over the time series data, writing each date and daily data
        for date, daily_data in sorted(time_series.items()):
            writer.writerow([
                date, 
                daily_data.get('1. open'), 
                daily_data.get('2. high'), 
                daily_data.get('3. low'), 
                daily_data.get('4. close'), 
                daily_data.get('5. volume')
            ])

for symbol in symbols:
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    
    # Save the fetched data to a CSV file for each symbol
    save_data_to_csv(symbol, data)

    # Notify the user
    print(f"Data for {symbol} saved to CSV.")
