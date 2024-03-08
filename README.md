# Stock Price Prediction using Data Mining in Financial Data

## Overview

This project aims to predict the future movement of stock prices for a selected set of stocks (e.g., IBM, JNJ, PG, V, and JPM) based on historical stock price data. The focus is on creating a predictive model that can forecast whether a stock's price will increase or decrease in the future.

## Problem Statement

### Predicting Stock Price Movement

#### Description

The problem at hand is to predict the future movement of stock prices for a selected set of stocks based on historical stock price data. The focus is on creating a predictive model that can forecast whether a stock's price will increase or decrease in the future.

#### Importance and Interest

1. **Investment Decision Making:** Accurate projections of stock price fluctuations are critical for investors, traders, and financial institutions when deciding whether to buy or sell stocks. A reliable model can help reduce risks and maximize returns.

2. **Financial Market Efficiency:** Understanding and anticipating stock price changes improve the overall efficiency of financial markets, ensuring that asset values represent all available data.

3. **Economic Measures:** Stock prices are often viewed as measures of economic health. Predicting stock price swings can help policymakers and analysts understand the overall economic environment and prospective developments.

4. **Risk Management:** Accurate projections lead to improved risk management techniques. Investors can mitigate potential losses by modifying their portfolios in response to expected market changes.

### Purpose of the Problem

1. **Complex Patterns:** Stock prices are impacted by a variety of factors such as market movements, company performance, economic data, and global events. Data science tools, such as machine learning, are well-suited to detecting complicated patterns in vast datasets, thereby capturing the dynamics of stock price fluctuations.

2. **Data Availability:** Historical stock price data is easily available, making it possible to apply data science solutions. Using libraries like 'yfinance', we can simply download and analyze data for analysis.

3. **Continuous Learning:** Stock markets are dynamic, and models must adjust to changing situations. Data science enables the development of machine learning models that can continuously learn and update predictions as new data becomes available.

## Data Collection and Processing

### Data Collection

The historical stock price data for the project is collected using the 'yfinance' library. The dataset includes five stocks: TSLA, JNJ, PG, V, and JPM, with a time period spanning five years from January 1, 2018, to January 1, 2023.

### Exploratory Data Analysis (EDA)

#### Summary Statistics

The dataset's basic information, summary statistics, and null values have been explored to gain insights into the data's structure and characteristics.

#### Data Exploration Visualizations

1. **Daily Returns:** Visualizing the daily returns of the selected stocks to understand their volatility.
2. **Weekly Returns:** Plotting weekly returns for a broader perspective on stock performance.
3. **Monthly Returns:** Exploring monthly returns to identify longer-term trends.

#### Additional Data Explorations

1. **Time Series Plot of Closing Prices:** Illustrating the time series plot of closing prices for different stocks.
2. **Volume of Each Stock:** Using boxplots to visualize the volume distribution for each stock.
3. **Closing Prices vs. Volume:** Creating a scatter plot to explore the relationship between closing prices and volume for all five stocks.

## Machine Learning Models

### Problem Formulation

The problem is formulated as a binary classification task, where the target variable is 1 for price increase and 0 for price decrease.

### Models Used

Several machine learning models are employed for prediction:

1. **Logistic Regression:** A linear model for binary classification.
2. **Random Forest Classifier:** An ensemble model combining multiple decision trees.
3. **Support Vector Machine:** Utilizes hyperplanes to classify data points.
4. **Decision Tree:** A tree-like model that predicts outcomes based on input features.
5. **K-Nearest Neighbors:** Classifies data points based on the majority class among their k-nearest neighbors.
6. **Naive Bayes:** A probabilistic model based on Bayes' theorem.
7. **AdaBoost:** An ensemble model that combines weak learners to form a strong learner.
8. **Gradient Boosting:** Builds a strong model by combining the predictions of multiple weak models.

### Model Evaluation

All models are trained on the data and evaluated using accuracy scores and classification reports.

### Visualizations and Summary

Confusion matrices and ROC curves are generated for model evaluation. The results highlight the models' performance in terms of precision, recall, and overall accuracy.

## Results and Conclusion

### Performance of Models

Based on the analysis of various machine learning models for predicting stock price movement, here is a summary of the results:

### Model Performance Summary:

| Model                     | Accuracy | Precision | Recall | F1-Score |
|---------------------------|----------|-----------|--------|----------|
| Logistic Regression       | 0.511    | 0.26      | 0.51   | 0.35     |
| Random Forest             | 0.539    | 0.54      | 0.54   | 0.54     |
| Support Vector Machine    | 0.511    | 0.26      | 0.51   | 0.35     |
| Decision Tree             | 0.509    | 0.51      | 0.51   | 0.51     |
| K-Nearest Neighbors       | 0.518    | 0.52      | 0.52   | 0.52     |
| Naive Bayes               | 0.511    | 0.26      | 0.51   | 0.35     |
| AdaBoost                  | 0.503    | 0.50      | 0.50   | 0.47     |
| Gradient Boosting         | 0.521    | 0.52      | 0.52   | 0.48     |

### Summary:

1. **Random Forest and Gradient Boosting:** These models outperform others in terms of accuracy, precision, recall, and F1-score. They demonstrate a better ability to capture the complexities in the data.

2. **Logistic Regression, Support Vector Machine, Naive Bayes, and AdaBoost:** These models show similar performance with accuracy around 50%. They may have limitations in capturing the underlying patterns in stock price movements.

3. **Decision Tree and K-Nearest Neighbors:** These models perform reasonably well but are not as robust as Random Forest and Gradient Boosting.

It's essential to consider the specific characteristics of the stock market and the dataset. Further tuning of hyperparameters or feature engineering might enhance model performance. Additionally, evaluating models on different time periods or with additional features could provide insights into their robustness and generalizability.

### Conclusion

While both models show modest performance, the Random Forest model outperforms Logistic Regression in terms of accuracy and overall balance between precision and recall. Further optimization and feature engineering may enhance the models' performance.

## Instructions for Running the Code

1. Install necessary libraries using `pip install -r requirements.txt`.
2. Run the Jupyter notebook `stock_price_prediction.ipynb` for step-by-step execution of the project.
# Stock-Market-Prediction
