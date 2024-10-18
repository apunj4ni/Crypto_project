# BTC Price Prediction with Sentiment Analysis

## Project Overview
This project aims to **predict Bitcoin’s future price** by combining two key factors:
1. **Historical Price Data**: How Bitcoin’s price has behaved over time.
2. **Public Sentiment**: How people feel about Bitcoin on social media platforms, specifically **Reddit** and **X (formerly Twitter)**.

The idea is that Bitcoin’s price is influenced by both **past trends** and **public opinion**, so merging these two sources of information can improve the accuracy of predictions.

---

## Project Workflow Summary

### 1. Collect Historical Data
- Use **Yahoo Finance** to gather **Bitcoin’s daily prices** and key metrics like **moving averages** and **volatility**.
- This data helps reveal **patterns** and **trends** in Bitcoin’s market behavior.

### 2. Gather Sentiment Data
- Pull **recent posts** about Bitcoin from **Reddit** and **X (Twitter)** using their APIs.
- Use **Natural Language Processing (NLP)** to analyze the sentiment of each post/tweet and classify it as **positive, negative, or neutral**.

### 3. Combine Both Data Sources
- **Merge the historical price data** and **daily sentiment scores** into a unified dataset.
- Example: If public sentiment is positive, the model might predict a **price increase**.

### 4. Build a Prediction Model
- Use **machine learning models** (e.g., Linear Regression, Random Forest, or LSTM) to predict Bitcoin’s **next-day closing price** based on historical trends and sentiment data.

### 5. Test the Model’s Accuracy
- **Evaluate the model** by comparing its predictions with actual prices.
- Use metrics like **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)** to assess how well the model performs.

### 6. Visualize the Results
- Create **graphs** comparing **predicted prices** with **actual prices** to visualize the model’s accuracy.
