# Stock Price Prediction and Forecasting Project

This repository contains a collection of scripts and data files for predicting and analyzing stock prices using machine learning models like Long Short-Term Memory (LSTM) networks and Seasonal AutoRegressive Integrated Moving Average (SARIMA). The project aims to provide automated trading insights and predictions based on historical stock prices.

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Project Overview

This project implements various machine learning models to forecast stock prices for a range of companies such as AMD, Amazon, Google, and others. The primary models included are:
- **LSTM Models**: For predicting future stock prices based on time-series data.
- **SARIMA Models**: For forecasting stock price trends by capturing seasonality and patterns in the data.

Additionally, there are scripts for extracting stock data, calculating daily returns, and working with multiple datasets.

## Repository Structure

```bash
├── AMD_stock_prices.csv               # AMD stock price data
├── AMZN_stock_prices.csv              # Amazon stock price data
├── ARI/                               # Directory containing additional resources and scripts for ARI-related analysis
├── COIN_stock_prices.csv              # Coinbase stock price data
├── GME_stock_prices.csv               # GameStop stock price data
├── GOOGL_stock_prices.csv             # Google stock price data
├── LSTM/                              # Directory with LSTM model-related code
│   └── LSTM_stock_price_model.py      # Example LSTM model script for stock price prediction
├── META_stock_prices.csv              # Meta (Facebook) stock price data
├── MSFT_stock_prices.csv              # Microsoft stock price data
├── MSTR_stock_prices.csv              # MicroStrategy stock price data
├── MU_stock_prices.csv                # Micron stock price data
├── NVDA_stock_prices.csv              # Nvidia stock price data
├── QQQ_stock_prices.csv               # QQQ ETF stock price data
├── SARIMA_SPY_Forecast.py             # SARIMA forecasting model for SPY stock prices
├── SMCI_5ltsm_prediction.py           # LSTM model for SMCI stock price prediction
├── SMCI_stock_prices.csv              # SMCI stock price data
├── SPY_stock_prices.csv               # SPY ETF stock price data
├── StockDataToCSV.py                  # Script for saving stock data to CSV
├── TSLA_stock_prices.csv              # Tesla stock price data
├── dailyreturns.py                    # Script to calculate daily returns
└── README.md                          # Project documentation (this file)

Key Scripts

	•	SARIMA_SPY_Forecast.py: Generates forecasts for SPY stock prices using SARIMA models.
	•	SMCI_5ltsm_prediction.py: Predicts SMCI stock prices using an LSTM model trained on historical data.
	•	StockDataToCSV.py: Extracts stock data and saves it as CSV files for further analysis.
	•	dailyreturns.py: Calculates daily percentage returns for the stocks in the dataset.

Data Files

The .csv files contain historical stock prices for various companies, which are used by the models for training and prediction.

Requirements

	•	Python 3.x
	•	Required libraries (install using pip):
	•	numpy
	•	pandas
	•	matplotlib
	•	scikit-learn
	•	tensorflow
	•	statsmodels (for SARIMA)

Installation

	1.	Clone the repository:
      git clone https://github.com/yourusername/your-repo.git
	2.	Navigate to the project directory:
      pip install -r requirements.txt
	3.	Install dependencies:
      pip install -r requirements.txt
	4.	Ensure you have the necessary stock data in the *.csv files or run StockDataToCSV.py to generate them.

Usage

	1.	To train and run the LSTM model for SMCI stock prediction, use:
                python SMCI_5ltsm_prediction.py
        2.	To forecast SPY stock prices using SARIMA, run:
                python SARIMA_SPY_Forecast.py
	3.	Use the dailyreturns.py script to calculate daily percentage returns for the stocks:
                python dailyreturns.py

StockDataToCSV.py

The StockDataToCSV.py script is designed to fetch daily stock price data from the Alpha Vantage API for a predefined list of stock symbols. The data is then saved into separate CSV files for each symbol. The script automates the process of data retrieval and storage, making it easier to work with historical stock data for analysis or model training.

How It Works:

	1.	API Request: The script uses the Alpha Vantage API to retrieve the daily time series stock data for a list of stock symbols, which currently includes companies like Intel (INTC), Tesla (TSLA), and Microsoft (MSFT).
	2.	Data Parsing: The script extracts relevant data such as the open, high, low, close prices, and trading volume from the API response.
	3.	CSV Output: The data is written to CSV files named after the stock symbols (e.g., INTC_stock_prices.csv, TSLA_stock_prices.csv). These files contain daily historical prices and volumes for each stock.

To run the script, ensure you have the necessary dependencies (requests, csv), and execute the file with: python StockDataToCSV.py

Footnote: The API key used in this script is for Alpha Vantage, which provides free API access for stock market data. Note that there is a limit on the number of API calls per minute in the free tier. If you encounter issues with the API, you may need to wait and rerun the script or consider upgrading to a premium API key.


License

This project is licensed under the MIT License. See the LICENSE file for details.
### Explanation:

- **Project Overview**: Provides an introduction to the project and its goals.
- **Repository Structure**: Outlines the files and directories in the project.
- **Requirements**: Lists the necessary libraries for running the project.
- **Installation**: Step-by-step guide to set up the project locally.
- **Usage**: Explains how to run specific scripts to achieve different outcomes (e.g., LSTM prediction, SARIMA forecasting).
- **License**: Details about the project's licensing.

Feel free to modify this based on the specific details of your project!




