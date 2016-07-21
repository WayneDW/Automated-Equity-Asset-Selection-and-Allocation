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

	#for num, ticker in sorted(enumerate(tickersList)):

	# randomize data
	rand_tickerList = tickersList.keys()
	random.shuffle(rand_tickerList)
	#print rand_tickerList
	for num, ticker in enumerate(rand_tickerList):
		if num + 1 > 50:
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
			model = class_data_processing.Fama_French_preprocessor(tickersList[ticker], start_time, end_time, \
				type_time, factor_num, rf, market_return, factors)
			# filter unqualified companies
			if not portfolio_filter.init_check(model, invest_type): continue
			results[ticker] = model.factor_regression()
			if not portfolio_filter.mid_check(results[ticker]): continue
			
		except:
			print ticker, "fetech failed, http error!"
			print "=============================================================================="
	return factors, results

def mean_covariance_estimation(start_time, end_time, type_time, factor_num, rf, invest_type):
	# parameters in covariane estimation by multi-factor model

	# I start a new function because the tickers list used before 
	# may not be the same with this one
	factors, results = multi_factor_regression(start_time, end_time, type_time, factor_num, rf, invest_type)

	# create numpy-format French factors 
	np_factors = np.empty([0, factor_num])
	for num, date in enumerate(sorted(factors)):
		# get return of the portfolio
		if factor_num == 3:
			np_factors = np.append(np_factors, np.array([[factors[date].Mkt_Rf, factors[date].SMB, \
				factors[date].HML]]), axis = 0)
		else:
			np_factors = np.append(np_factors, np.array([[factors[date].Mkt_Rf, factors[date].SMB, \
				factors[date].HML, factors[date].RMW, factors[date].CMA]]), axis = 0)

	# subcovariance estimation
	sub_cov =  np.cov(np.transpose(np_factors), ddof = 1)
	
	# for comparison with CAPM
	#sub_cov =  np.matrix([[0.0012442,0,0],[0,0,0],[0,0,0]])
	#print sub_cov


	output = {'ticker':[], 'return':[], 'VaR':[]}
	betas = np.empty([0, factor_num])

	for ticker in results:
		output['ticker'].append(ticker)
		output['return'].append(results[ticker].expect_return)
		output['VaR'].append(results[ticker].VaR)

		betas = np.append(betas, np.array([results[ticker].betas]), axis = 0)

	# covariance = betas * sub_cov * betas', where betas is a N*L matrix, L is the factor_num
	covariance = np.dot(betas, sub_cov)
	covariance = np.dot(covariance, np.transpose(betas))

	# Due to error accumulation, the current covariance can't use its own sigma to derive 
	# the correlation coefficient to be 1, we need to modify the diagonal of the covariance
	for num, ticker in enumerate(output['ticker']):
			covariance[num, num] = results[ticker].var

	return output, np.matrix(covariance)
