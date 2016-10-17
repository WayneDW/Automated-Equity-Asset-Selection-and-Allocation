# This repository is sperated into four parts:

## 1. Data Preprocessing

```python
./crawl.py
```

crawl ticker list and basic stock from http://www.nasdaq.com/screening/company-list.aspx

crawl financial statement from http://financials.morningstar.com/ratios/r.html?t=BIDU&region=USA&culture=en_US

crawl historical stock prices from  https://finance.yahoo.com/

## 2. Feature engineering

```python
./feature engineering.py
```
### 2.1 Feature format

		Basic feature filter to delete useless features

		format basic information to the (sample_n, feature_m) matrix

	    [ feature1_sample_1, feature2_sample_1, ... feature_m_sample_1]

	    ...

	    [ feature1_sample_m, feature2_sample_m, ... feature_m_sample_n]

	    Typically, a sample has feature dimension 83 * 11 (financial ratios)

### 2.2 Missing value filter

	    Delele the feature if it has more than N% missing values (we can set N as 1, 5, 10)

### 2.3 feature time-horizon completeness check

		Since we need a complete time window to shift, if it doesn't have full 11 data, delete all of this type of feature, such P/E or Asset turnover

### 2.4 Feature date filter

		Some stock may only have data from 2005 - 2014

		Some starts at April 2007, we regard them as 2006

### 2.5 Feature variance filter

		Right now, this part is complete, just have been annotated



## 3. Stock classification based on financial statements

```python
./learning.py
```

### 3.1 Train the model

Take financial ratios (2006 - Dec.2014) to train the model

Label based on Sortino ratio and CVaR with time between Jan, 2014 and Jan, 2015

Select best parameters for us.

### 3.2 Predict data and make comparison

Predict the performance based on financial ratios (2007 - Dec.2015)

Check if the selected group is sigficant better than the rest
	



```
Comment: We wse SVM in sklearn to train the model and use cross-validation to pick the best parameters (missing value problem, now is dealed with deleting missing values, may be handled by cubic spline in the future)
```

Get a roughly desired stock sets


## 4. Optimize the best weight for your portfolio

```python
./mean_variance_optimization.py
```

Do a brute-force method to randomly pick 15 stocks from the stock sets and implement mean-variance portfolio with no short constraint

Delelte the portfolio with the worst CVAR 



