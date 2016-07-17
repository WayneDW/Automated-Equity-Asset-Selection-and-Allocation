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


# to capture key results from the following model

class ticker_detail:
	def __init__(self, r, var, alpha, betas, ticker_info):
		self.expect_return = r
		self.var = var
		self.alpha = alpha
		self.betas = betas
		# ticker_info is an object, containing info: Mkt-cap, IPO year,...
		self.ticker_info = ticker_info 


class Fama_French_factor_model:
	def __init__(self, ticker_info, start_time, end_time, type_time, factor_num, rf, market_return, factors):
		self.ticker_info = ticker_info
		self.ticker = ticker_info.ticker
		self.start_time = start_time
		self.end_time = end_time
		self.type_time = type_time
		self.factor_num = factor_num
		self.rf = rf
		self.market_return = market_return
		self.factors = factors

		spider = class_crawler.web_spider()
		self.returns = spider.yahoo_return(self.ticker, start_time, end_time, type_time)

	def factor_regression(self):
		# form pandas dataframe
		pars = {'date':[],'return':[], 'Mkt_Rf':[], 'SMB':[], 'HML':[], 'Excess_Return':[]} 
		if self.factor_num == 5:
			pars['RMW'] = []; pars['CMA'] = []

		for num, date in enumerate(sorted(self.returns)):
			if date not in self.factors:
				print "French factors haven't update to " + date
				break
			if num == 0: print "Starting time:", date
			if num == len(self.returns) - 1: print "Ending   time:", date
			pars['date'].append(date)
			pars['return'].append(self.returns[date].adjClose)
			pars["Mkt_Rf"].append(self.factors[date].Mkt_Rf)
			pars["SMB"].append(self.factors[date].SMB)
			pars["HML"].append(self.factors[date].HML)
			pars["Excess_Return"].append(self.returns[date].adjClose - self.factors[date].RF)
			if self.factor_num == 5:
				pars["RMW"].append(self.factors[date].RMW)
				pars["CMA"].append(self.factors[date].CMA)
		pd.DataFrame(pars)
		#pd.DataFrame(pars, index = dates, columns = ["Mkt_Rf", "SMB", "HML", "Excess_Return"])
		# We can get the 3 risk premiums from: Equity Asset Evaluation, 2nd edition by Jerald Pinto, etc. page 66.
		# annulized risk premiums is Market-rf: 5.5%, Size: 2%, Value: 4.3%
		# or from the average of the French's factors over the time period
		if self.factor_num == 3:
			model = ols("Excess_Return ~ Mkt_Rf + SMB + HML", pars).fit()
			# The following 3 numbers are from: Equity Asset Evaluation, 2nd edition by Jerald Pinto, etc. page 66.
			market_risk_coef, size_risk_coef, value_risk_coef = model.params[1:]
			required_rate_of_return = self.rf + market_risk_coef * np.mean(pars["Mkt_Rf"]) \
			+ size_risk_coef * np.mean(pars["SMB"]) + value_risk_coef * np.mean(pars["HML"])
		else:
			model = ols("Excess_Return ~ Mkt_Rf + SMB + HML + RMW + CMA", pars).fit()
			market_risk_coef, size_risk_coef, value_risk_coef, profit_risk_coef, invest_risk_coef = model.params[1:]
			required_rate_of_return = self.rf + market_risk_coef * np.mean(pars["Mkt_Rf"]) \
			+ size_risk_coef * np.mean(pars["SMB"]) + value_risk_coef * np.mean(pars["HML"])
			+ profit_risk_coef * np.mean(pars["RMW"]) + invest_risk_coef * np.mean(pars["CMA"])

		print model.summary()
		# take care!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		alpha = model.params[0]
		betas = model.params[1:]
		var = np.var(np.array(pars['return']), ddof = 1)
		print "alpha :", round(alpha, 3)
		print "Return:", util.perc(required_rate_of_return, 3)
		print "SD    :", util.perc(math.sqrt(var), 2), "\n\n"
		#print "annualized required_rate_of_return:", required_rate_of_return * 12
		return ticker_detail(required_rate_of_return, var, alpha, betas, self.ticker_info)

		#fig, ax = plt.subplots(figsize=(12, 8))
		#fig = sm.graphics.influence_plot(model, ax = ax, criterion = "cooks")
		#plt.show()




