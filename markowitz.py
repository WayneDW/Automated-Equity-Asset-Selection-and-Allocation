import math
import numpy as np

import util
import class_crawler
import fama_french_factor_model
from fama_french_factor_model import mean_covariance_estimation


# For a tangency portfolop
# L = w' Cov w + lamda [w'u - rf], taking derivative...
# <=> wp = Inv(Cov) * u / 1' Inv(Cov) u

def mean_variance_portfolio(start_time, end_time, type_time, factor_num, invest_type):
	spider = class_crawler.web_spider()
	# fetch treasury yield data from Bloomberg
	treasuryList = spider.treasury_yield()
	rf = treasuryList["GB6"] / 12
	returns, covariance = mean_covariance_estimation(start_time, end_time, type_time, factor_num, rf, invest_type)

	dim = len(returns['ticker'])
	r_minus_rf = np.subtract(returns['return'], [rf] * dim)

	# top_vec = inv(covariance) * r <=> 
	# solve top_vec from covariance * top_vec = r_minus_rf
	top_vec = np.linalg.solve(covariance, r_minus_rf)

	# test the solution of the result
	precision = np.dot(covariance, top_vec) - r_minus_rf
	print "Precision in solving linear systems", np.linalg.norm(precision), max(precision)
	bot_num = np.dot(top_vec, [1] * dim)

	# portfolio key stat: weight, return, variance
	weight = top_vec / bot_num
	portfolio_return = np.dot(returns['return'], weight)
	portfolio_variance = np.dot(np.dot(weight, covariance), weight)
	print "Portfolio Summary"
	print "=============================================================================="
	print "Portfolio   size: ", len(weight)
	print "Portfolio   name: ", returns['ticker']
	print "Portfolio weight: ", util.array_perc(weight, 1)
	print "Portfolio return: ", util.perc(portfolio_return, 2)
	try:
		print "Portfolio  SD(Y): ", util.perc(math.sqrt(portfolio_variance), 2)
	except:
		print "------------------------------math domain error,", portfolio_variance
	print "Annulized return: ", util.perc(portfolio_return * 12, 2)
	print "Annulized  SD(Y): ", util.perc(math.sqrt(portfolio_variance * 12), 2)
	print "        SP ratio: ", round(portfolio_return / math.sqrt(portfolio_variance) * math.sqrt(12),2)
	return returns['ticker'], weight



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




