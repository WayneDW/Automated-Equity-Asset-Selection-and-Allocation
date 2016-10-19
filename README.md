# This repository is separated into four parts:

## 1. Data Preprocessing

```python
./crawl.py
output: ./input/raw_*
```


crawl ticker list and basic stock from http://www.nasdaq.com/screening/company-list.aspx

crawl financial statement from http://financials.morningstar.com/ratios/r.html?t=BIDU&region=USA&culture=en_US

crawl historical stock prices from  https://finance.yahoo.com/

## 2. Feature engineering


```python
input: feature_projection
./feature engineering.py
output: feature_label_*, selected_feature_*
```


### 2.1 Feature format

Format basic information to the (sample_n, feature_m) matrix

		[ feature1_sample_1, feature2_sample_1, ... feature_m_sample_1]

		...

		[ feature1_sample_m, feature2_sample_m, ... feature_m_sample_n]

Before filter, a sample has feature dimension 81 * 11 (891 financial ratios)

### 2.2 Missing value filter

Delele the feature if it has more than N% missing values (we can set N as 1, 5, 10)

### 2.3 feature time-horizon completeness check

Since we need a complete time window to shift, if it doesn't have full 11 data, delete all of this 

type of feature, such P/E or Asset turnover

### 2.4 Feature date filter

Some stock may only have data from 2005 - 2014

Some starts at April 2007, we regard them as 2006

## 3. Data accuracy check

```python
./dataCheck.py
```

## 4. Stock classification based on financial statements

```python
./learning.py
```

### 4.1 Train the model

Take financial ratios (2006 - Dec.2014) to train the model

Increase training sample by using 2006-2011, 2007-2012, 2008-2013, 2009-2014, 2010-2015

Label based on Sortino ratio

### 4.2 Predict data and make comparison

In summary, gradient boosting gives us the best performance (highest precision of label 1 in average)
	


## 5. Optimize the best weight for your portfolio

```python
./mean_variance_optimization.py
```

Do a brute-force method to randomly pick 15 stocks from the stock sets and implement mean-variance portfolio with no short constraint

Delelte the portfolio with the worst CVAR 



