import math
import numpy as np
from scipy.optimize import minimize

import util
import class_crawler
import fama_french_factor_model
from fama_french_factor_model import mean_covariance_estimation

# constraint optimization is to limit the size of short sale

# http://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
class const_markowitz:
	def __init__(self, start_time, end_time, type_time, factor_num, invest_type):
		spider = class_crawler.web_spider()
		# fetch treasury yield data from Bloomberg
		treasuryList = spider.treasury_yield()
		rf = treasuryList["GB6"] / 12
		returns, covariance = mean_covariance_estimation(start_time, end_time, type_time, \
			factor_num, rf, invest_type)
		dim = len(returns['ticker'])
		self.tickerList = returns['ticker']
		self.returns = returns['return']
		self.VaR = returns['VaR']
		self.covariance = covariance
		self.rf = rf
		self.dim = dim

	# objective function, maximize sharpe ratio <=> minimize -sharpe ratio
	def minFunc(self, weight, sign = -1):
		r_minus_rf = np.dot(weight, self.returns) - self.rf
		sigma = math.sqrt(np.dot(np.dot(weight, self.covariance), weight)[0, 0])
		return sign * (r_minus_rf / sigma) * 3.464

	def derivaFunc(self, weight, sign = -1):
		r_minus_rf = np.dot(weight, self.returns) - self.rf
		sigma = math.sqrt(np.dot(np.dot(weight, self.covariance), weight)[0, 0])

		cov_w = np.dot(self.covariance, weight)
		result = sign * (np.subtract(np.array(self.returns) / sigma, r_minus_rf * cov_w / sigma ** 3))
		return np.array(result)

	def const_optimization(self, lowBound, upperBound):
		'''
		cons = ( {'type': 'eq', \
				  'fun' : lambda weight: np.array([np.dot(weight, [1] * self.dim) - 1]), \
			      'jac' : lambda weight: np.array(weight)}, \
				 {'type': 'ineq', \
			      'fun' : lambda weight: np.array(min(weight) + 0.2), \
			      'jac' : lambda weight: np.array([1] * self.dim)}, \
			      {'type': 'ineq', \
			      'fun' : lambda weight: np.array(1 - max(weight)), \
			      'jac' : lambda weight: np.array([-1] * self.dim)})
		'''
		R_threshold = np.percentile(self.VaR, 95)
		print "Return threshold in constraint: ", R_threshold
		cons = ({'type': 'eq', \
				  'fun': lambda weight: np.array([np.dot(weight, [1] * self.dim) - 1]), \
			      'jac': lambda weight: np.array(weight)},
			    {'type': 'ineq',\
			     'fun' : lambda weight: np.array([np.dot(self.VaR, weight) - R_threshold]), \
			     'jac' : lambda weight: np.array(self.VaR)})
		init = [1.0 / self.dim] * self.dim
		bnds = zip([lowBound] * self.dim, [upperBound] * self.dim)
		opts = {'disp': True,  'maxiter': 1e3, 'ftol': 1e-16} 
		res = minimize(self.minFunc, init, args = (-1,), constraints = cons, \
			jac = self.derivaFunc, bounds = bnds, method = 'SLSQP', options = opts)

		weight = res.x
		portfolio_return = np.dot(self.returns, weight)
		portfolio_variance = np.dot(np.dot(weight, self.covariance), weight)
		print "Portfolio Summary"
		print "=============================================================================="
		print "Portfolio   size:", len(weight)
		print "Portfolio   name:", self.tickerList
		print "Portfolio weight:", util.array_perc(weight, 1)
		print "Portfolio return:", util.perc(portfolio_return, 2)
		try:
			print "Portfolio  SD(Y): ", util.perc(math.sqrt(portfolio_variance), 2)
		except:
			print "------------------------------math domain error,", portfolio_variance
		print "Annulized return:", util.perc(portfolio_return * 12, 2)
		print "Annulized  SD(Y):", util.perc(math.sqrt(portfolio_variance * 12), 2)
		print "        SP ratio:", round(portfolio_return / math.sqrt(portfolio_variance) * math.sqrt(12),2)
		print "Return threshold:", util.perc(R_threshold, 2)
		return self.tickerList, weight



