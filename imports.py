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

yesterday = datetime.now() - timedelta(1)
formatted_date = yesterday.strftime('%Y-%m-%d')
print(formatted_date)
btc_ticker = yf.Ticker("BTC-USD")
print(btc_ticker.history(start="2017-09-17", end=formatted_date))
