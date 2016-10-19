import os
import re
import numpy as np
import scipy as sp

from scipy.interpolate import interp1d
from sklearn import linear_model
from sklearn.preprocessing import scale
from sklearn.preprocessing import Imputer
from collections import OrderedDict
from collections import Counter

# this script need a large amount of memory
# haven't consider the industry information and marketCap

class preprocessing:
	def __init__(self, date_start, date_end, date_type):
		self.loadDict() # load key_ratio list to filter other ratios
		self.readFile() # read file to load features and label information
		self.createFeature() # create feature matrix and get price information
		self.interpolate() # fill in missing value with interpolated value
		self.featureName() # add feature name w.r.t. the columns of feature matrix
		self.calculateRisk() # get each stock returns, CVaR, downside sd
		self.createLabel() # create labels corresponding to features based on price
		self.cleanFeatures() # delete feature if it contains too many missing values
		self.saveLocal() # save ticker_feature_label matrix to local
		
	def loadDict(self):
		print "Load feature dictionary"
		f = open('./input/feature_projection')
		self.wordDict = {}
		self.wordDictRev = {}
		for num, line in enumerate(f):
			feature, ftype = line.strip().split('\t')
			self.wordDict[feature] = ftype
			self.wordDictRev[num] = feature

	def readFile(self):
		print "Read Raw Data"
		f = open('./input/' + "_".join(["raw", date_start, date_end, date_type]))
		res = ""
		for i, line in enumerate(f):
			res += line
		self.lists = re.findall(r"([\w\d\-_%*&/]+:[\w\d\.\-/ ]+)", res) # match key-value pair

	def createFeature(self):
		print "Create Feature Matrix"
		# turn the feature in a (samples, features) matrix, A is a feature for one sample
		self.feature, A = np.empty([0, 11 * 80]), np.empty([0, 11])
		self.tickerList = [] # use array, since we need the location of each ticker
		self.stock_prices = {}# 2-D hash table
		last, cnt, cnt_del = -1, 1, 1
		for _, line in enumerate(self.lists): # read key-value pairs
			#if _ > 5 * 10 ** 6: break # used to test large file
			dat = line.strip().split(":") # split key-value pair
			if dat[0] == "ticker": # everytime update new ticker, clear past array
				ticker = dat[1]
				# in year 2016, most data starts in 2006
				# the rest focus on 2007 (they just IPO in 2007, so 2007 has 2 data)
				if last != -1 and years[0] in ['2006', '2007'] and np.shape(A) == (80, 11):
					A = A.flatten() # change 2-D matrix to 1-D
					self.tickerList.append(last)
					self.feature = np.vstack([self.feature, A])
					print cnt, last; cnt += 1
				
				A = np.empty([0, 11]) # reinitialize A
				self.stock_prices[ticker] = {} # initialize stock price
				last = ticker
			if dat[0] == "key_ratios_Time":
				years = np.array(dat[1].split(' '))
				if len(years) == 11: self.time_horizon = years

			if dat[0].startswith("key_ratios_"): 
				for currency in ["USD", "CNY", "GBP", "JPY", "EUR", "CAD", "CHF", "AUD", "HKD"]:
					dat[0] = dat[0].replace("_" + currency, "") # unify currency
				if dat[0] in self.wordDict: # if key_ratios are these we need
					numList = np.array(dat[1].split(' ')) # create one row in numpy
					if len(numList) != np.shape(A)[1]: continue # in case format not match
					A = np.vstack([A, numList]) # add to A

			m = re.search(r"(\d{4}\-\d{2}\-\d{2})_adjClose", dat[0]) # match stock price
			if m:
				curDate = m.group(1)
				stock_price = float(dat[1])
				self.stock_prices[ticker][curDate] = stock_price
			
		# add the last qualified sample
		if years[0] in ['2006', '2007'] and np.shape(A) == (80, 11):
			if last != -1 and len(self.stock_prices[last]) > 252 * 3: # data check
				A = A.flatten() # change 2-D matrix to 1-D
				self.tickerList.append(last)
				self.feature = np.vstack([self.feature, A])
		self.feature = self.feature.astype(np.float)

	def interpolate(self): # handle missing values (middle of data)
		# model can't handle missing value (despite Boosting could) 
		# we will move forward time window to make prediction       
		
		print "Interpolate and predict missing values"
		regr = linear_model.LinearRegression()
		
		for kth, dataSets in enumerate(self.feature):
			for num, item in enumerate(dataSets):
				if num % 11 != 0: continue # operate every 11 loops
				x = np.array(range(num, num + 11))
				y = dataSets[num: num + 11]
				newX = x[~np.isnan(y)] # ignore nan value
				newY = y[~np.isnan(y)]
				if len(newX) < 5 or len(newX) == 11:
					continue # data complete or too many missing values

				if max(newX) - min(newX) + 1 != len(newX): # missing value between min and max
					grid_x = np.linspace(min(newX), max(newX), max(newX) - min(newX) + 1)
					self.feature[kth][min(newX): max(newX) + 1] = sp.interpolate.interp1d(
						newX, newY, kind='cubic')(grid_x).round(4)
					
				if max(newX) - min(newX) + 1 != 11: # linear regression required
					partX = np.linspace(min(newX), max(newX), max(newX) - min(newX) + 1)
					partY = self.feature[kth][min(newX): max(newX) + 1]
					partX, partY = partX.reshape(len(partX), 1), partY.reshape(len(partY), 1)
					regr.fit(partX, partY)
					for r_ in range(num, num + 11):
						if r_ >= min(newX) and r_ <= max(newX): continue
						self.feature[kth][r_] = round(regr.predict(r_), 4)

	def featureName(self):
		self.feature_name = []
		for num in self.wordDictRev:
			len_time = len(self.time_horizon)
			for k, time in enumerate(self.time_horizon):			
				self.feature_name.append(self.wordDictRev[num].split("key_ratios_")[1] \
					+ "__" + time)
		self.feature_name = np.array(self.feature_name)
	
	def calculateRisk(self):
		print "Compute stock risk ratios"
		self.returns, self.DR, self.SD = {}, {}, {} # DR: downside deviation
		self.CVAR = {95:{}, 99:{}, 99.9:{}}
		for ticker in self.stock_prices:
			prices = self.stock_prices[ticker]
			self.returns[ticker] = {}
			orderedDt = OrderedDict(sorted(prices.items())) # sort map by key
			for num, date in enumerate(orderedDt):
				if num == 0:
					last = date
				else:
					if prices[last] > 0:
						self.returns[ticker][date] = prices[date] / prices[last] - 1
					last = date
			returns_risk = np.array(self.returns[ticker].values())
			returns_risk = returns_risk[~np.isnan(returns_risk)] # delete missing value
			self.DR[ticker] = np.std(returns_risk[returns_risk <= 0]) * np.sqrt(252)
			# for alpha in [0.1, 1, 5]: # change to percentile alpha
			# 	VaR = np.percentile(returns_risk, alpha)
			# 	if len(returns_risk[returns_risk < VaR]) == 0: continue
			# 	#self.CVAR[100 - alpha][ticker] = np.mean(returns_risk[returns_risk < VaR])
			# 	self.DR[ticker] = np.std(returns_risk[returns_risk <= 0]) * np.sqrt(252)
			# 	#self.SD[ticker] = np.std(returns_risk) * np.sqrt(252) # annualized
			
	def createLabel(self):
		print "Create Label Based on Sortino ratio"
		self.label = {}
		for name in ["train0", "train1", "train2", "train3", "test"]:
			self.label[name] = np.zeros(len(self.feature)) # to train model
		if len(self.feature) != len(self.tickerList): 
			sys.exit("feature number doesn't match label information")

		risk_free = 0.016 # USD LIBOR - 12 months
		def calcRatio(date0, date1, type, ticker, mul):
			if date0 in self.stock_prices[ticker] and date1 in self.stock_prices[ticker]:
				annualized_r = self.stock_prices[ticker][date1] / \
							self.stock_prices[ticker][date0] - 1			
				# Sortino ratio is better to evalueate high-volatility portfolio
				Sortino_ratio = (annualized_r - risk_free) / self.DR[ticker] * mul
				#Sharpe_ratio = (annualized_r - risk_free) / self.SD[ticker] * mul
				self.label[type][_] = round(Sortino_ratio, 1)
			else:
				self.label[type][_] = np.nan
		for _ in xrange(len(self.tickerList)):
			ticker = self.tickerList[_]
			calcRatio("2012-01-03", "2012-03-01", "train0", ticker, 252. / 40)# 40 days
			calcRatio("2013-01-02", "2013-03-01", "train1", ticker, 252. / 40) # 40 days
			calcRatio("2014-01-02", "2014-03-03", "train2", ticker, 252. / 40) # 40 days
			calcRatio("2015-01-02", "2015-03-03", "train3", ticker, 252. / 40) # 40 days
			calcRatio("2016-01-04", "2016-03-02", "test", ticker, 252. / 40) # 40 days

	def cleanFeatures(self):
		print "\nClean Features"
		# this part should not include ticker info as its 1st col
		print "Raw feature dimension: ", np.shape(self.feature)
		tag_none_ratio = np.repeat(True, np.shape(self.feature)[1])
		threshold = 0.05 # missing value threshold, 0 is too rigid
		for num in range(np.shape(self.feature)[1]):
			features_j = self.feature[:, num]
			# compute the ratio of missing value in feature_j
			none_ratio = float(len(features_j[np.isnan(features_j)])) / len(features_j)
			if none_ratio > threshold: tag_none_ratio[num] = False
		self.feature = self.feature[:,tag_none_ratio]
		self.feature_name = self.feature_name[tag_none_ratio]
		print('Feature dimension after %d%%-missing-value check: %s' \
			% (threshold * 100, np.shape(self.feature)))

		# tag true with feature variance is above than threshold, otherwise tag false
		#print len((np.std(self.feature[~np.isnan(self.feature)], axis=0) > 0))
		#tag_sd = (np.std(self.feature.astype(np.float), axis=0) > 0) # this can't deal with nan
		# tag_sd = np.repeat(True, np.shape(self.feature)[1])
		# for num in range(np.shape(self.feature)[1]):
		# 	std = np.std(self.feature[~np.isnan(self.feature[:,num]), num])
		# 	#mean = np.mean(self.feature[~np.isnan(self.feature[:,num]), num])
		# 	if std == 0: tag_sd[num] = False
		# self.feature = self.feature[:,tag_sd]
		# self.feature_name = self.feature_name[tag_sd]
		# print "Feature dimension after variance check: ", np.shape(self.feature)

		# if a ratio in features doesn't have 11 data, we can't get time-window shift
		# delete all the related ratios asscociated with that
		tag_full_ratio = np.repeat(True, np.shape(self.feature)[1])
		last = -1; cnt = 0
		for num in range(len(self.feature_name)):
			tag = self.feature_name[num].split("__")[0]
			if last != -1 and tag != last:
				if cnt != 11:
					tag_full_ratio[num - cnt:num] = np.repeat(False, cnt)
				cnt = 0
			cnt += 1
			last = tag
		if tag != last and cnt != 11:
			tag_full_ratio[num - cnt:num] = np.repeat(False, cnt)
		self.feature = self.feature[:,tag_full_ratio]
		self.feature_name = self.feature_name[tag_full_ratio]
		print 'Feature dimension after ratio-time completeness check:', \
			np.shape(self.feature)

		# do data imputation to fill in missing values
		# load file of feature_name to determine which feature to scale
		imp = Imputer(missing_values='NaN', strategy='median', axis=0)
		imp.fit(self.feature)
		self.feature = imp.transform(self.feature)

		# feature scaling, can't do it in createFeature part due to missing values
		fclasses = len(self.feature[0]) / 11
		for num in range(len(self.feature)):
			feature_trans = np.reshape(self.feature[num], [fclasses, 11])
			#feature_trans = scale(feature_trans, axis=1).round(3)
			# we should scale the non-percent values
			for fnum in range(fclasses):
				featureName = "key_ratios_" + self.feature_name[fnum].split("__")[0]
				featureType = self.wordDict[featureName]
				if featureType == "scalar": # transform to scale number
					feature_trans[fnum] = scale(feature_trans[fnum]).round(3)
				elif featureType == "percent": # transform xy% to 0.xy
					feature_trans[fnum] = feature_trans[fnum] / 100.
				else: continue # actually, we should not go into this loop
			self.feature[num] = feature_trans.flatten()

		# add ticker to the 1st col, label to the last col of the feature matrix
		self.tickerList = np.array(self.tickerList)
		self.feature = self.feature.transpose()
		self.feature = np.vstack([self.tickerList, self.feature])
		for name in ["train0", "train1", "train2", "train3", "test"]:
			self.feature = np.vstack([self.feature, self.label[name]])
		self.feature = self.feature.transpose()
		print "Feature dimension after col-merge: ", np.shape(self.feature)

		# if the stock don't have coresponding price, delete it.
		# tag_price = np.repeat(True, len(self.feature))
		# for num in xrange(len(self.feature)):
		# 	if self.feature[num][0] in self.deleteList:
		# 		tag_price[num] = False
		# self.feature = self.feature[tag_price]
		# print "Feature dimension after price check: ", np.shape(self.feature)


	def saveLocal(self):
		np.savetxt("./input/feature_label_" + date_start + '_' + date_end, \
			self.feature, delimiter=',', fmt="%s")

		np.savetxt("./output/selected_feature_" + date_start + '_' + date_end, \
			self.feature_name, delimiter=',', fmt="%s")



if __name__ == "__main__":
	date_start = "2000-01-01"
	date_end = "2016-12-31"
	date_type = "d"
	s = preprocessing(date_start, date_end, date_type)