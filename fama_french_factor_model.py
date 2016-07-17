#!/usr/bin/python
import re
import sys
import numpy as np
import pandas as pd
import operator
import math
import itertools
import random
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy import stats

from pandas.tools import plotting
from statsmodels.formula.api import ols
from sklearn.decomposition import PCA

import util
import class_crawler
import class_data_processing
import portfolio_filter


# par3: m->month, w->week, d->day, factor in [3, 5]
def multi_factor_regression(start_time, end_time, type_time, factor_num, rf, invest_type):
	# create a web_spider object to fetch data
	spider = class_crawler.web_spider()

	# fetch tickers List from NASDAQ, an object containing basic ticker info
	tickersList = spider.tickers()

	# fetch factors data from French's website
	factors = spider.FF_factor(factor_num)

	# calculate the mean return of the market portfolio, namely S&P 500
	market_returns = spider.yahoo_return("^GSPC", start_time, end_time,type_time)
	market_return = 0; dim = len(market_returns)
	for date in market_returns:
		market_return += market_returns[date].adjClose / dim
	results = {}

	#testCase = ['CBG','VET','SIX','NWL','COO','NKE','DEI','JNJ','CINF','HAS','WM','ECA','LUX','FITB','SRC']
	for num, ticker in sorted(enumerate(tickersList)):
		if num + 1 > 30:
			break
		print "\n\nTicker: " + str(num + 1)
		print "=============================================================================="
		print "Ticker       |\t", ticker
		print "Company      |\t", tickersList[ticker].name
		print "Market Cap   |\t", tickersList[ticker].marketCap
		print "IPO year     |\t", tickersList[ticker].IPOyear
		print "Sector       |\t", tickersList[ticker].sector
		print "Industry     |\t", tickersList[ticker].industry
		print "Exchange     |\t", tickersList[ticker].exchange
		print "=============================================================================="
		try:
			model = class_data_processing.Fama_French_factor_model(tickersList[ticker], start_time, end_time, \
				type_time, factor_num, rf, market_return, factors)
			# filter unqualified companies
			if not portfolio_filter.check(model, invest_type): continue
			results[ticker] = model.factor_regression()
			
		except:
			print ticker, "fetech failed, http error!"
	return factors, results


