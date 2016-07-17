import math
import numpy as np

import util
import class_crawler
import fama_french_factor_model
from fama_french_factor_model import multi_factor_regression


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
	print sub_cov
	
	# for comparison with CAPM
	#sub_cov =  np.matrix([[0.0012442,0,0],[0,0,0],[0,0,0]])
	#print sub_cov


	returnsList = {'ticker':[], 'return':[]}
	betas = np.empty([0, factor_num])

	for ticker in results:
		returnsList['ticker'].append(ticker)
		returnsList['return'].append(results[ticker].expect_return)
		betas = np.append(betas, np.array([results[ticker].betas]), axis = 0)

	# covariance = betas * sub_cov * betas', where betas is a N*L matrix, L is the factor_num
	covariance = np.dot(betas, sub_cov)
	covariance = np.dot(covariance, np.transpose(betas))

	# Most importantly, due to error accumulation, the current covariance can't use its own sigma
	# to derive the correlation coefficient to be 1, we need to modify the diagonal of the covariance
	for num, ticker in enumerate(returnsList['ticker']):
			covariance[num, num] = results[ticker].var

	return returnsList['ticker'], returnsList, np.matrix(covariance)

# For a tangency portfolop
# L = w' Cov w + lamda [w'u - rf], taking derivative...
# <=> wp = Inv(Cov) * u / 1' Inv(Cov) u

def mean_variance_portfolio(start_time, end_time, type_time, factor_num, invest_type):
	spider = class_crawler.web_spider()
	# fetch treasury yield data from Bloomberg
	treasuryList = spider.treasury_yield()
	rf = treasuryList["GB6"] / 12
	tickerList, returns, covariance = mean_covariance_estimation(start_time, end_time, type_time, factor_num, rf, invest_type)

	dim = len(returns['ticker'])
	r_minus_rf = np.subtract(returns['return'], [rf] * dim)

	# top_vec = inv(covariance) * r <=> 
	# solve top_vec from covariance * top_vec = r_minus_rf
	top_vec = np.linalg.solve(covariance, r_minus_rf)
	bot_num = np.dot(top_vec, [1] * dim)

	# portfolio key stat: weight, return, variance
	weight = top_vec / bot_num
	portfolio_return = np.dot(returns['return'], weight)
	portfolio_variance = np.dot(np.dot(weight, covariance), weight)
	print "Portfolio Summary"
	print "=============================================================================="
	print "Portfolio   size: ", len(weight)
	print "Portfolio   name: ", tickerList
	print "Portfolio weight: ", util.array_perc(weight, 1)
	print "Portfolio return: ", util.perc(portfolio_return, 2)
	try:
		print "Portfolio  SD(Y): ", util.perc(math.sqrt(portfolio_variance), 2)
	except:
		print "------------------------------math domain error,", portfolio_variance
	print "Annulized return: ", util.perc(portfolio_return * 12, 2)
	print "Annulized  SD(Y): ", util.perc(math.sqrt(portfolio_variance * 12), 2)
	print "        SP ratio: ", round(portfolio_return / math.sqrt(portfolio_variance) * math.sqrt(12),2)
	return tickerList, weight

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




