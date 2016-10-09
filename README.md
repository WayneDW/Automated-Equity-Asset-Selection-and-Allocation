# This repository is sperated into four parts:

## Data Preprocessing

crawl ticker list and basic stock from http://www.nasdaq.com/screening/company-list.aspx

crawl financial statement from http://www.morningstar.com/

crawl historical stock prices from  https://finance.yahoo.com/

```python
./crawl.py
```

## Feature engineering

format basic information to the (sample_n, feature_m) matrix

[ feature1_sample_1, feature2_sample_1, ... feature_m_sample_1]

...

[ feature1_sample_m, feature2_sample_m, ... feature_m_sample_n]

```python
./feature engineering.py
```

Basic feature filter to delete useless features

Take financial ratios as input and sharpe ratios as label

## Stock classification based on stock price


Use SVM in sklearn to train the model and use cross-validation to pick the best parameters (missing value problem, maybe handled by cubic spline)

Get a roughly desired stock sets

```python
./learning.py
```
## Optimize the best weight for your portfolio

Do a brute-force method to randomly pick 15 stocks from the stock sets and implement mean-variance portfolio with no short constraint

Delelte the portfolio with the worst CVAR 


```python
./mean_variance_optimization.py
```
