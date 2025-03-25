# Bitcoin Price and Returns Prediction using Machine Learning

## Introduction

#### This repository contains an end to end predictive analysis with two distinct projects aimed at predicting Bitcoin prices and returns using machine learning techniques. 
**It first starts with the 2014-2024 Bitcoin Data Analysis which was done for data cleaning ,feature engineering etc to get the additional columns in the dataset used for machine learning**
**The first project employs a Random Forest Regression model to predict the Bitcoin price target variable. The second project extends this approach by utilizing both Random Forest Regression and Gradient Boosting Regression models to predict both Bitcoin price and daily returns target variables.**
**The primary objective of these projects is to assess the predictive performance of these models on financial time series data and identify the most effective approach for forecasting Bitcoin price movements and returns.**


## Data Preparation

### Dataset Description

**The dataset utilized for both projects is sourced from "Bitcoin_Historical_Data.csv" on Kaggle Datasets, which contains historical Bitcoin price data and was later feature enginneered to add more columns like technical indicators, 
daily returns etc. using Microsoft excel and python. ("Bitcoin_Historical_Data1.csv") .**
**Both before and after datasets are avalibale in the repository**

**Some Key Columns Include:**

 • date: Timestamp of the data point
 
 • adj_close: Adjusted closing price
 
 • close: Closing price (used as the price target variable)
 
 • high: Highest price in the period
 
 • low: Lowest price in the period
 
 • open: Opening price
 
 • volume: Trading volume
 
 • daily returns: Daily percentage change in price (used as the returns target variable)
 
 • Various technical indicators (e.g., MA50, MA200, RSI, MACD, etc.)

## Data Cleaning and Preprocessing

### The preprocessing workflow begins with loading the dataset and performing initial cleaning steps:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

• **Load the dataset** :
```
df = pd.read_csv('Bitcoin_Historical_Data1.csv', parse_dates=['date'])
print(df.head())
```

• **Date Formatting** : The date column is converted to a datetime format with UTC timezone for consistency:
```
df['date'] = pd.to_datetime(df['date'], utc=True)
```

• **Column Selection**: Non-numeric columns (except date) are dropped to focus on numerical features suitable for modeling:
```
dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
coltokeep = [col for col in df.columns if col.lower() == 'date' or df[col].dtype.name in dtypes]
df1 = df[coltokeep]
print("Columns retained:", df1.columns.tolist())
```
• **Handling Missing Values**: Columns with NaN values (e.g., MA50, MA200, RSI) are identified. The dataset contains missing values due to the calculation of technical indicators requiring a lookback period. For simplicity, forward fill is applied to impute missing values, though mean imputation could be an alternative depending on the context:
```
nan_columns = df1.columns[df1.isna().any()].tolist()
print("Columns with NaN entries:", nan_columns)
df1.fillna(method='ffill', inplace=True)
```
• **Feature Scaling** : Features are standardized to ensure uniform scale across variables, which is critical for model performance:

```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df1.drop(['date', 'close', 'daily returns'], axis=1))

```

• **Train-Test Split** : The data is split into training and testing sets using an 80-20 split, respecting the time series nature of the data:

```
train_size = 0.8
split_idx = int(len(df1) * train_size)
X_train = X_scaled[:split_idx]
X_test = X_scaled[split_idx:]
y_price_train = df1['close'].iloc[:split_idx]
y_price_test = df1['close'].iloc[split_idx:]
y_returns_train = df1['daily returns'].iloc[:split_idx]
y_returns_test = df1['daily returns'].iloc[split_idx:]

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_price_train shape:", y_price_train.shape)
print("y_price_test shape:", y_price_test.shape)
print("y_returns_train shape:", y_returns_train.shape)
print("y_returns_test shape:", y_returns_test.shape)

```

**The resulting shapes indicate a training set of 2977 samples and a testing set of 744 samples, with 55 features after preprocessing.**



## Exploratory Data Analysis

**To understand the target variables, distributions are visualized**:

```
plt.figure(figsize=(15, 5))

# Returns distribution
plt.subplot(1, 2, 1)
sns.histplot(df1['daily returns'], bins=50)
plt.title('Distribution of Returns')
plt.axvline(x=0, color='r', linestyle='--')

# Price distribution
plt.subplot(1, 2, 2)
sns.histplot(df1['close'], bins=50)
plt.title('Distribution of Prices')

plt.tight_layout()
plt.show()

print("\nTarget Statistics:")
print("Returns Mean:", df1['daily returns'].mean())
print("Returns Std Dev:", df1['daily returns'].std())
print("Returns Skewness:", df1['daily returns'].skew())
print("Prices Mean:", df1['close'].mean())
print("Prices Std Dev:", df1['close'].std())
print("Prices Skewness:", df1['close'].skew())

```

 • **Returns** : Mean ≈ 0.0021, Std Dev ≈ 0.0363, Skewness ≈ -0.1151 (near-symmetric with slight negative skew)
 
 • **Prices** : Mean ≈ 18814.01, Std Dev ≈ 20809.71, Skewness ≈ 1.1327 (right-skewed due to price growth over time)

 **A correlation matrix of scaled features is also plotted to identify relationships**:

```
plt.figure(figsize=(12, 8))
sns.heatmap(pd.DataFrame(X_scaled, columns=df1.drop(['date', 'close', 'daily returns'], axis=1).columns).corr(), annot=False, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Scaled Features')
plt.show()

```

## Model Training

### Project 1: Random Forest Regression for Price Prediction

**The first project focuses solely on predicting Bitcoin prices using a Random Forest Regression model**:

```
from sklearn.ensemble import RandomForestRegressor

rf_pmodel = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_pmodel.fit(X_train, y_price_train)
```

### Project 2: Random Forest and Gradient Boosting for Price and Returns Prediction

**The second project extends the analysis to predict both price and daily returns using two models**:

```
from sklearn.ensemble import GradientBoostingRegressor

# Random Forest models
rf_rmodel = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_rmodel.fit(X_train, y_returns_train)
# rf_pmodel already trained above

# Gradient Boosting models
gb_rmodel = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_pmodel = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

gb_rmodel.fit(X_train, y_returns_train)
gb_pmodel.fit(X_train, y_price_train)
```

 • Hyperparameters: Both models use 100 trees/estimators. Random Forest uses a max depth of 10, while Gradient Boosting uses a learning rate of 0.1. These are initial settings; further tuning could enhance performance.

## Model Evaluation

**Model performance is assessed using three metrics: R-squared (R2), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE)**:

```
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

def evaluate_model(y_true, y_pred, model_name):
    metrics = {
        'R2': r2_score(y_true, y_pred),
        'RMSE': mean_squared_error(y_true, y_pred, squared=False),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred)
    }
    print(f"\n{model_name} Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    return metrics
```

#### Predictions
```
rf_returns_pred = rf_rmodel.predict(X_test)
gb_returns_pred = gb_rmodel.predict(X_test)
rf_price_pred = rf_pmodel.predict(X_test)
gb_price_pred = gb_pmodel.predict(X_test)
```

#### Evaluate models
```
rf_returns_metrics = evaluate_model(y_returns_test, rf_returns_pred, 'Random Forest Returns')
gb_returns_metrics = evaluate_model(y_returns_test, gb_returns_pred, 'Gradient Boosting Returns')
rf_price_metrics = evaluate_model(y_price_test, rf_price_pred, 'Random Forest Price')
gb_price_metrics = evaluate_model(y_price_test, gb_price_pred, 'Gradient Boosting Price')
```        

# RESULTS :

### RETURNS 

  **Random Forest Returns Metrics** :
  
 • R2: 0.8098
 
 • RMSE: 0.0110
 
 • MAPE: 8.3519
 
  **Gradient Boosting Returns Metrics** :
 
 • R2: 0.7731
 
 • RMSE: 0.0120
 
 • MAPE: 9.1903

 

### PRICE

  **Random Forest Price Metrics** :
  
 • R2: 0.9411
 
 • RMSE: 4713.5885
 
 • MAPE: 0.0348

 
  **Gradient Boosting Price Metrics** :
  
 • R2: 0.9429
 
 • RMSE: 4640.6441
 
 • MAPE: 0.0358


**ANALYSIS** :

 • **Price Prediction**:  Both models excel, with R2 scores above 0.94 and low MAPE values (~3.5%), indicating high accuracy. Gradient Boosting slightly outperforms Random Forest with a lower RMSE.

 • **Returns Prediction**:  Performance is moderate, with R2 scores of 0.77-0.81. Random Forest outperforms Gradient Boosting, with a higher R2 and lower error metrics, though MAPE values (8-9%) suggest challenges in capturing return volatility.


## VISUALIZATIONS

###  Predictions vs. Actual Values

**Predictions are visualized to compare model outputs against actual data**:

```
plt.figure(figsize=(15, 10))
```

### Returns predictions
```
plt.subplot(2, 1, 1)
plt.plot(y_returns_test.index, y_returns_test, label='Actual Returns', alpha=0.7)
plt.plot(y_returns_test.index, rf_returns_pred, label='RF Predicted Returns', alpha=0.7)
plt.plot(y_returns_test.index, gb_returns_pred, label='GB Predicted Returns', alpha=0.7)
plt.title('Returns Predictions vs Actual')
plt.legend()
```

### Price predictions
```
plt.subplot(2, 1, 2)
plt.plot(y_price_test.index, y_price_test, label='Actual Price', alpha=0.7)
plt.plot(y_price_test.index, rf_price_pred, label='RF Predicted Price', alpha=0.7)
plt.plot(y_price_test.index, gb_price_pred, label='GB Predicted Price', alpha=0.7)
plt.title('Price Predictions vs Actual')
plt.legend()

plt.tight_layout()
plt.show()
```

 • **Returns Plot**:  Both models capture the general trend but struggle with high volatility periods, showing deviations from actual values.

 • **Price Plot**:  Predictions closely align with actual prices, reflecting the high R2 scores and confirming robust forecasting ability.

## Feature Importance

### Feature importance is analyzed to identify key predictors:
```
def plot_feature_importance(model, feature_names, title):
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.barh(importance['feature'], importance['importance'])
    plt.title(title)
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()
```

### Plot feature importance
```
plot_feature_importance(rf_rmodel, df1.drop(['date', 'close', 'daily returns'], axis=1).columns, 'Feature Importance - Returns Prediction (RF)')
plot_feature_importance(rf_pmodel, df1.drop(['date', 'close', 'daily returns'], axis=1).columns, 'Feature Importance - Price Prediction (RF)')
plot_feature_importance(gb_rmodel, df1.drop(['date', 'close', 'daily returns'], axis=1).columns, 'Feature Importance - Returns Prediction (GB)')
plot_feature_importance(gb_pmodel, df1.drop(['date', 'close', 'daily returns'], axis=1).columns, 'Feature Importance - Price Prediction (GB)')
```

 • **Key Features :  Technical indicators like MA50, MA200, RSI, and volume consistently rank high, underscoring their importance in predicting both price and returns.**

## Conclusion

### Summary of Findings

 • **Project 1 (Price Prediction with Random Forest):**
 The Random Forest Regression model effectively predicts Bitcoin prices, achieving an **R2 of 0.9411** and a low **MAPE of 0.0348** , indicating strong predictive capability for price levels.

 • **Project 2 (Price and Returns Prediction):**
 
 • **Price Prediction** : Both Random Forest and Gradient Boosting models perform exceptionally  well, with Gradient Boosting slightly edging out with an ** R2 of 0.9429** and **RMSE of 4640.6441.**

 • **Returns Prediction** : **Random Forest (R2: 0.8098)**  outperforms** Gradient Boosting (R2: 0.7731)**, though both models face challenges in capturing the full volatility of returns, as evidenced by higher MAPE values.

## Insights

 •** Price prediction**  is more robust than returns prediction, likely due to the smoother trends in price data compared to the noisy, volatile nature of daily returns.

 • **Technical indicators**  play a critical role in model performance, suggesting that feature engineering is a key driver of success.
 

### Recommendations and Future Work
• Hyperparameter Tuning: Optimize n_estimators, max_depth, and learning_rate using grid search or random search to improve model performance.

 • Feature Expansion: Incorporate external data, such as sentiment analysis from news or social media, to enhance returns prediction.
 
 • Advanced Models: Explore deep learning approaches (e.g., LSTM networks) to better capture temporal dependencies in the data.
 
 • Missing Value Handling: Experiment with alternative imputation methods (e.g., interpolation) to assess their impact on model accuracy.

## Model Saving

Trained models are saved for future use:
```
from joblib import dump

dump(gb_pmodel, 'GBpricemodel.joblib')
dump(gb_rmodel, 'GBreturnsmodel.joblib')
dump(rf_pmodel, 'RFpricemodel.joblib')
dump(rf_rmodel, 'RFreturnsmodel.joblib')
```	
Alternatively, models are also saved using pickle:
```
import pickle

with open('GBpricemodel.pkl', 'wb') as file:
    pickle.dump(gb_pmodel, file)
with open('GBreturns.pkl', 'wb') as file:
    pickle.dump(gb_rmodel, file)
with open('RFpricemodel.pkl', 'wb') as file:
    pickle.dump(rf_pmodel, file)
with open('RFreturns.pkl', 'wb') as file:
    pickle.dump(rf_rmodel, file)
```

## Usage

To replicate this project:

 1. Ensure all dependencies (pandas, numpy, matplotlib, seaborn, sklearn, joblib, pickle) are installed.
 2. Place Bitcoin_Historical_Data1.csv in the working directory.
 3. Run the provided code snippets in sequence, adjusting paths as necessary.

This README provides a comprehensive guide to the workflow, ensuring transparency and reproducibility of the Bitcoin price and returns prediction projects.
