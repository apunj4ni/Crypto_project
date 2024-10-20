
import pandas as pd
import numpy as np
import requests
#import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

yesterday = datetime.now() - timedelta(1)
formatted_date = yesterday.strftime('%Y-%m-%d')
print(formatted_date)
btc_ticker = yf.Ticker("BTC-USD")
print(btc_ticker.history(start="2017-09-17", end=formatted_date))
btc_data = btc_ticker.history(start="2017-09-17", end=formatted_date)
# Feature engineering: Add moving averages and volume change
btc_data['SMA_20'] = btc_data['Close'].rolling(window=20).mean()
btc_data['SMA_50'] = btc_data['Close'].rolling(window=50).mean()
btc_data['Volume_change'] = btc_data['Volume'].pct_change()
btc_data = btc_data.dropna()

# Features (X) and Target (y)
X = btc_data[['Close', 'SMA_20', 'SMA_50', 'Volume_change']]
y = btc_data['Close'].shift(-1).dropna()
X = X[:-1]  # Align features with target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Dictionary to store models and their results
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
}

# Function to train and evaluate models
def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    model_results = {}
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict and calculate metrics
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        model_results[name] = {
            'model': model,
            'mse': mse,
            'mae': mae,
            'y_pred': y_pred
        }
        print(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}")
    
    return model_results

# Train and evaluate models
results = train_and_evaluate(models, X_train, y_train, X_test, y_test)


def plot_predictions(results, y_test, title="Model Predictions"):
    plt.figure(figsize=(12, 8))
    plt.plot(y_test.values, label='Actual', color='black', linewidth=2)
    
    for name, result in results.items():
        plt.plot(result['y_pred'], label=f'{name} Predictions')
    
    plt.title(title)
    plt.xlabel("Days")
    plt.ylabel("Bitcoin Price")
    plt.legend()
    plt.show()

plot_predictions(results, y_test)


# Resample Bitcoin data to get the last price of each month
btc_data_monthly = btc_data.resample('M').last()

# Feature engineering on monthly data
btc_data_monthly['SMA_20'] = btc_data_monthly['Close'].rolling(window=2).mean()  # Rolling mean over 2 months (as an example)
btc_data_monthly['SMA_50'] = btc_data_monthly['Close'].rolling(window=5).mean()
btc_data_monthly['Volume_change'] = btc_data_monthly['Volume'].pct_change()

btc_data_monthly = btc_data_monthly.dropna()

# Features (X) and Target (y) on monthly data
X_monthly = btc_data_monthly[['Close', 'SMA_20', 'SMA_50', 'Volume_change']]
y_monthly = btc_data_monthly['Close'].shift(-1).dropna()
X_monthly = X_monthly[:-1]  # Align features with target

# Split into training and testing sets
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_monthly, y_monthly, test_size=0.2, random_state=42)
# Train and evaluate models on monthly data
results_monthly = train_and_evaluate(models, X_train_m, y_train_m, X_test_m, y_test_m)
plot_predictions(results_monthly, y_test_m, title="Model Predictions (Monthly)")

#os.makedirs(sys.path[0]+)x
