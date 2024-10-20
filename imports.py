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
import praw

yesterday = datetime.now() - timedelta(1)
formatted_date = yesterday.strftime('%Y-%m-%d')
print(formatted_date)
btc_data = btc_ticker.history(start="2017-09-17", end=formatted_date) # Store in btc_data variable for future use
print(btc_data)

# Adding 50-day and 200-day SMA
btc_data['MA_50'] = btc_data['Close'].rolling(window = 50).mean()
btc_data['MA_200'] = btc_data['Close'].rolling(window = 200).mean()

# Add SD of returns
btc_data['Daily_Return'] = btc_data['Close'].ptc_change()
btc_data['Volatility_30d'] = btc_data['Daily_Return'].rolling(window = 30).std()

# Show updated data frame
print(btc_data.head())

# Plot closing price and MAs for visualization
plt.figure(figsize = (12, 6))
sns.lineplot(data=btc_data, x=btc_data.index, y='Close', label='Closing Price')
sns.lineplot(data=btc_data, x=btc_data.index, y='MA_50', label='50-day Moving Average')
sns.lineplot(data=btc_data, x=btc_data.index, y='MA_200', label='200-Day Moving Average')
plt.title('Bitcoin Closing Price and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Initialize Reddit API with credentials
reddit = praw.reddit(
    client_id = 'fKBzn113qbMMGvJEtOvtBQ',
    client_secret = 'mIFoOyzgLd0AH85BdhOGmUH8o-Qafg',
    user_agent = 'BTC Sentiment Analyzer by /u/Leadership-Think' #Reddit username
)

# Fetch recent posts from r/BitcoinMarkets
posts = reddit.subreddit('BitcoinMarkets').new(limit = 30) # Reddit API limit is 60 for free accounts. Can be increased later
for post in posts:
    print(post.title)
    print(post.author)
    print(post.score)
    print(post.selftext)