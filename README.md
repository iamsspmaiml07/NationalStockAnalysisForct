# StockBuddy Assistant (NSE/BSE)

**StockBuddy Assistant** is an intuitive web application built using **Streamlit** that enables users to fetch, visualize, and forecast stock market data. It supports stock symbols from the **NSE (National Stock Exchange)** and **BSE (Bombay Stock Exchange)**, and uses **yfinance** to fetch historical stock data and a **Linear Regression model** to forecast stock prices for the next 7 days.

## Features

### Fetch Real-Time Stock Data
- Input any stock symbol for **NSE** (e.g., `RELIANCE.NS`) or **BSE** (e.g., `TCS.BO`).
- Fetches daily stock data using **yfinance**.

### Visualize Stock Trends
- Interactive line charts display stock prices including **Open**, **High**, and **Closing** prices.
- Visualizations help analyze stock price movements over time.

### Historical Data Table
- A clean, tabular representation of the stock's historical data for analysis.
- Data includes **Open**, **High**, and **Closing** prices.

### 1-Week Stock Price Forecast
- Utilizes a **Linear Regression** model to forecast the stock's closing prices for the next 7 days.
- Forecasted prices are appended to the historical data for easy comparison.

### Investment Recommendation
- Based on the forecasted data, the app provides actionable recommendations:
  - **BUY**: If the stock shows an upward trend over the forecasted period.
  - **HOLD**: If no significant upward trend is detected.

### User-Friendly Interface
- Intuitive input fields and buttons for easy data interaction.
- Fully responsive interface compatible with desktops and tablets.

## How to Use

### Run the Application
Ensure all required Python libraries are installed (see **Installation** section below). Start the app with the following command:

```bash
streamlit run app.py
