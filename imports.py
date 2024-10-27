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
btc_ticker = yf.Ticker("BTC-USD")
btc_data = btc_ticker.history(start="2017-09-17", end=formatted_date) # Store in btc_data variable for future use
print(btc_data)

# Adding 50-day and 200-day SMA
btc_data['MA_50'] = btc_data['Close'].rolling(window = 50).mean()
btc_data['MA_200'] = btc_data['Close'].rolling(window = 200).mean()

# Add SD of returns
btc_data['Daily_Return'] = btc_data['Close'].pct_change()
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
reddit = praw.Reddit(
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

# Sentiment analyzer 
sentiment_analyzer = SentimentIntensityAnalyzer()
'''
# Calculate VADER sentiment score
def get_textblob_sentiment(texts):
    scores = [TextBlob(text).sentiment.polarity for text in texts]
    return np.mean(scores)
'''

reddit_texts = [post.title + " " + post.selftext for post in posts]

def get_vader_sentiment(texts):
    scores = [sentiment_analyzer.polarity_scores(text)['compound'] for text in texts]
    return np.mean(scores)
    
reddit_sentiment = get_vader_sentiment(reddit_texts)
# x_sentiment = get_vader_sentiment(x_texts) 

print(f'Reddit Sentiment Score: {reddit_sentiment:.3f}')
# print(f'X Sentiment Score: {x_sentiment:.3f}')

# Merge Sentiment Scores with BTC price
sentiment_data = pd.DataFrame ({
    'Date': [formatted_date],
    'Reddit Sentiment': [reddit_sentiment]
    # 'X Sentiment': [x_sentiment]
})

# Merge Sentiment Data with BTC Data
btc_data.reset_index(inplace=True)
btc_data['Date'] = btc_data['Date'].astype(str)

btc_data = pd.merge(btc_data, sentiment_data, on = 'Date', how = 'left')

# Default empty sentiment data with 0
btc_data.fillna(0, inplace=True)

print(btc_data.head())
