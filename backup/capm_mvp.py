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

import spider_FF


# to capture key results from the following model
class key_index:
	def __init__(self, r, var, alpha, betas):
		self.expect_return = r
		self.var = var
		self.alpha = alpha
		self.betas = betas

class Fama_French_factor_model:
	def __init__(self, start_time, end_time, type_time, ticker, factor_num, treasuryList, factors):
		self.start_time = start_time
		self.end_time = end_time
		self.type_time = type_time
		self.factor_num = factor_num
		#tickerlist = spider.tickers()
		self.treasuryList = treasuryList
		self.factors = factors

		spider = spider_FF.web_spider()
		self.returns = spider.yahoo_return(start_time, end_time, ticker, type_time)
		#for l in sorted(factors):
		#	print l, factors[l]

	def factor_regression(self, market_return):
		# form pandas dataframe
		pars = {'date':[],'return':[], 'Mkt_Rf':[], 'SMB':[], 'HML':[], 'Excess_Return':[]} 
		if self.factor_num == 5:
			pars['CMA'] = []; pars['RMW'] = []

		for num, date in enumerate(sorted(self.returns)):
			if date not in self.factors:
				print "French factors haven't update to " + date
				break
			if num == 0: print "Starting time: ", date
			if num == len(self.returns) - 1: print "Ending   time:", date
			pars['date'].append(date)
			pars['return'].append(self.returns[date].adjClose)
			pars["Mkt_Rf"].append(self.factors[date].Mkt_Rf)
			pars["SMB"].append(self.factors[date].SMB)
			pars["HML"].append(self.factors[date].HML)
			pars["Excess_Return"].append(self.returns[date].adjClose - self.factors[date].RF)
			if self.factor_num == 5:
				pars["CMA"].append(self.factors[date].CMA)
				pars["RMW"].append(self.factors[date].RMW)
		pd.DataFrame(pars)
		#pd.DataFrame(pars, index = dates, columns = ["Mkt_Rf", "SMB", "HML", "Excess_Return"])
		if self.factor_num == 3:
			model = ols("Excess_Return ~ Mkt_Rf", pars).fit()
			print model.params
			market_risk_coef = model.params[1]
			market_risk_premium = 0.0727/12
			required_rate_of_return = self.treasuryList["GB6"] / 12 + market_risk_coef * market_risk_premium
		else:
			model = ols("Excess_Return ~ Mkt_Rf + SMB + HML + CMA + RMW", pars).fit()
			# the other two risk premiums come from the average of the French's factors over the 1990 - now
			market_risk_premium, size_risk_premium, value_risk_premium, profit_risk_premium, \
			invest_risk_premium = [0.055 / 12, 0.02 / 12, 0.043 / 12, 0.041 / 12, 0.03 / 12]
			market_risk_coef, size_risk_coef, value_risk_coef, profit_risk_coef, invest_risk_coef = model.params[1:]
			required_rate_of_return = self.treasuryList["GB6"] / 12 + market_risk_coef * market_risk_premium \
			+ size_risk_coef * size_risk_premium + value_risk_coef * value_risk_premium \
			+ profit_risk_coef * profit_risk_premium + invest_risk_coef * invest_risk_premium
		print model.summary()
		alpha = model.params[0]
		betas = model.params[1]
		#alpha = np.mean(np.array(pars['return'])) - model.params[1] * market_return
		var = np.var(np.array(pars['return']))
		print "alpha: ", alpha
		##########################################################################################################tmp
		print "beta", betas
		var = np.var(pars['return'])
		#print model.params
		print "Required_rate_of_return:", required_rate_of_return

		#print "annualized required_rate_of_return:", required_rate_of_return * 12
		return key_index(required_rate_of_return, var, alpha, [betas, 0, 0])

		#fig, ax = plt.subplots(figsize=(12, 8))
		#fig = sm.graphics.influence_plot(model, ax = ax, criterion = "cooks")
		#plt.show()


# par3: m->month, w->week, d->day, factor in [3, 5]
def multi_factor_regression(start_time, end_time, type_time, factor_num):
	# create a web_spider object to fetch data
	spider = spider_FF.web_spider()

	# fetch tickers List from NASDAQ
	tickersList = spider.tickers()

	# fetch treasury yield data from Bloomberg
	treasuryList = spider.treasury_yield()
	print "treasuryList", treasuryList

	# fetch factors data from French's website
	factors = spider.FF_factor(factor_num)

	# calculate the mean return of the market portfolio, namely S&P 500
	market_returns = spider.yahoo_return(start_time, end_time, "^GSPC",type_time)
	market_return = 0; dim = len(market_returns)
	for date in market_returns:
		market_return += market_returns[date].adjClose / dim

	results = {}
	testList = ['CBG','VET', 'SIX',	'NWL',	'COO',	'NKE',	'DEI',	'JNJ',	'CINF',	'HAS',	'WM',	'ECA',	'LUX',	'FITB',	'SRC']
	#for num, ticker in sorted(enumerate(tickersList)):
	for num, ticker in sorted(enumerate(testList)):
		#if num > 10:
		#	break
		print "\n\n" + str(num)
		print "****************************************************"
		print "Ticker       |\t", ticker
		print "Company      |\t", tickersList[ticker].name
		print "Market Cap   |\t", tickersList[ticker].marketCap
		print "IPO year     |\t", tickersList[ticker].IPOyear
		print "Sector       |\t", tickersList[ticker].sector
		print "Industry     |\t", tickersList[ticker].industry
		print "****************************************************"
		if 1:
			model = Fama_French_factor_model(start_time, end_time, type_time, ticker, factor_num, treasuryList, factors)
			results[ticker] = model.factor_regression(market_return)
	return factors, results


def mean_covariance_estimation(start_time, end_time, type_time, factor_num):
	# parameters in covariance estimation by multi-factor model

	# I start a new function because the tickers list used before 
	# may not be the same with this one
	factors, results = multi_factor_regression(start_time, end_time, type_time, factor_num)

	# create numpy-format French factors 
	np_factors = np.empty([0, factor_num])
	for num, date in enumerate(sorted(factors)):
		# get return of the portfolio
		if factor_num == 3:
			np_factors = np.append(np_factors, np.array([[factors[date].Mkt_Rf, factors[date].SMB, \
				factors[date].HML]]), axis = 0)
		else:
			np_factors = np.append(np_factors, np.array([factors[date].Mkt_Rf, factors[date].SMB, \
				factors[date].HML, actors[date].CMA, actors[date].RMW]), axis = 0)

	# subcovariance estimation
	#sub_cov =  np.cov(np.transpose(np_factors))
	# in test, only one factor
	sub_cov =  np.matrix([[0.0012442,0,0],[0,0,0],[0,0,0]])

	returnsList = {'ticker':[], 'return':[]}
	betas = np.empty([0, factor_num])
	testList = ['CBG','VET', 'SIX',	'NWL',	'COO',	'NKE',	'DEI',	'JNJ',	'CINF',	'HAS',	'WM',	'ECA',	'LUX',	'FITB',	'SRC']
	#for ticker in results:
	for ticker in testList:
		returnsList['ticker'].append(ticker)
		returnsList['return'].append(results[ticker].expect_return)
		betas = np.append(betas, np.array([results[ticker].betas]), axis = 0)

	# covariance = betas * sub_cov * betas', where betas is a N*L matrix, L is the factor_num
	covariance = np.dot(betas, sub_cov)
	covariance = np.dot(covariance, np.transpose(betas))

	# Most importantly, due to error accumulation, the current covariance can't use its own sigma
	# to derive the correlation coefficient to be 1, we need to modify the diagonal of the covariance
	for num, ticker in enumerate(testList):
		print covariance[num, num]
		covariance[num, num] = results[ticker].var
	print "------------------------------------------------------------------------\n", covariance
	return returnsList, np.matrix(covariance)



# For a tangency portfolop
# L = w' Cov w + lamda [w'u - rf], taking derivative...
# <=> wp = Inv(Cov) * u / 1' Inv(Cov) u

def mean_variance_portfolio(start_time, end_time, type_time, factor_num):
	returns, covariance = mean_covariance_estimation(start_time, end_time, type_time, factor_num)
	Rf = 0.00016
	dim = len(returns['ticker'])

	r_minus_rf = np.subtract(returns['return'], [Rf] * dim)

	# top_vec = inv(covariance) * r <=> 
	# solve top_vec from covariance * top_vec = r_minus_rf
	top_vec = np.linalg.solve(covariance, r_minus_rf)
	bot_num = np.dot(top_vec, [1] * dim)

	# portfolio key stat: weight, return, variance
	weight = top_vec / bot_num
	
	portfolio_return = np.dot(returns['return'], weight)
	protfolio_variance = np.dot(np.dot(weight, covariance), weight)
	print "weight:", weight
	print "Annulized return:", (1 + portfolio_return) ^ 12 - 1
	print "Annulized SD    :", math.sqrt(protfolio_variance * 12)
	print "Sharp Ratio     :", ((1 + portfolio_return) ^ 12 - 1 - Rf * 12) / math.sqrt(protfolio_variance * 12)

def portfolio_pca(start_time, end_time, type_time, factor_num):
	mean, cov = mean_covariance_estimation(start_time, end_time, type_time, factor_num)
	# Virtually, PCA is SVD
	U, D, V = np.linalg.svd(cov, full_matrices=True)
	

	total_var = sum(D)
	threshold = 0.95
	acumulate = 0
	for num in range(len(D)):
		acumulate += D[num]
		# the last part suffers the problem of numerial instability
		if acumulate > threshold:
			break

	mean_trans = np.transpose(mean['return'])
	# R = inv(U) * r <=> U*R = r, we solve this linear equation
	return_principle_component = np.linalg.solve(U, mean_trans)




def main():
	mean_variance_portfolio("2011-02-20","2016-02-01", "m", 3)

if __name__ == "__main__":
	sys.exit(main())


