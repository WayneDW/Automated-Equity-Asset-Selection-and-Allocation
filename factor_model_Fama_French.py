#!/usr/bin/python
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy import stats

from pandas.tools import plotting
from statsmodels.formula.api import ols

import spider_FF



class Fama_French_factor_model:
	def __init__(self, start_time, end_time, type_time, ticker, factor_num):
		self.start_time = start_time
		self.end_time = end_time
		self.type_time = type_time
		self.factor_num = factor_num
		spider = spider_FF.web_spider()
		#tickerlist = spider.tickers()
		self.treasuryList = spider.treasury_yield()
		self.factors = spider.FF_factor(factor_num)
		self.returns = spider.yahoo_return(start_time, end_time, ticker, type_time)
		#for l in sorted(factors):
		#	print l, factors[l]

	def factor_regression(self):
		# form pandas dataframe
		pars = {'Mkt_Rf':[], 'SMB':[], 'HML':[], 'Excess_Return':[]} 
		if self.factor_num == 5:
			pars['CMA'] = []; pars['RMW'] = []
		dates = []
		for date in sorted(self.returns):
			dates.append(date)
			print date
			if date not in self.factors:
				print "French factors haven't update to " + date
				break
			pars["Mkt_Rf"].append(self.factors[date].Mkt_Rf)
			pars["SMB"].append(self.factors[date].SMB)
			pars["HML"].append(self.factors[date].HML)
			pars["Excess_Return"].append(self.returns[date].adjClose - self.factors[date].RF)
			if self.factor_num == 5:
				pars["CMA"].append(self.factors[date].CMA)
				pars["RMW"].append(self.factors[date].RMW)

		pd.DataFrame(pars)

		#print pd.DataFrame(pars, index = dates, columns = ["Mkt_Rf", "SMB", "HML", "Excess_Return"])
		if self.factor_num == 3:
			model = ols("Excess_Return ~ Mkt_Rf + SMB + HML", pars).fit()
			# The following 3 numbers are from: Equity Asset Evaluation, 2nd edition by Jerald Pinto, etc. page 66.
			market_risk_premium, size_risk_premium, value_risk_premium = [0.055 / 12, 0.02 / 12, 0.043 / 12]
			market_risk_coef, size_risk_coef, value_risk_coef = model.params[1:]
			required_rate_of_return = self.treasuryList["GT10:GOV"] / 12 + market_risk_coef * market_risk_premium \
			+ size_risk_coef * size_risk_premium + value_risk_coef * value_risk_premium
		else:
			model = ols("Excess_Return ~ Mkt_Rf + SMB + HML + CMA + RMW", pars).fit()
			# the other two risk premiums come from the average of the French's factors over the 1990 - now
			market_risk_premium, size_risk_premium, value_risk_premium, profit_risk_premium, \
			invest_risk_premium = [0.055 / 12, 0.02 / 12, 0.043 / 12, 0.041 / 12, 0.03 / 12]
			market_risk_coef, size_risk_coef, value_risk_coef, profit_risk_coef, invest_risk_coef = model.params[1:]
			required_rate_of_return = self.treasuryList["GT10:GOV"] / 12 + market_risk_coef * market_risk_premium \
			+ size_risk_coef * size_risk_premium + value_risk_coef * value_risk_premium \
			+ profit_risk_coef * profit_risk_premium + invest_risk_coef * invest_risk_premium
		print model.summary()
		alpha = model.params[0]
		
		print model.params
		print "required_rate_of_return:", required_rate_of_return * 12

		fig, ax = plt.subplots(figsize=(12, 8))
		fig = sm.graphics.influence_plot(model, ax = ax, criterion = "cooks")
		plt.show()
		




def main():
	print "Owesome!"
	model = Fama_French_factor_model( "2010-09-01", "-",'m', "BIDU", 3)
	model.factor_regression()

if __name__ == "__main__":
	sys.exit(main())


