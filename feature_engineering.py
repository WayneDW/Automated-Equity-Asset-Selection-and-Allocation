import os
import re
import collections
import numpy as np

class preprocessing:
	def __init__(self, date_start, date_end, date_type):
		self.d0, self.d1, self.dtype = date_start, date_end, date_type
		self.loadDict() # load key_ratio list to filter other ratios
		self.readFile() # read file to load features and label information
		self.createFeature() # create feature matrix and get price information
		self.createLabel() # create labels corresponding to features based on price
		self.cleanFeatures() # delete feature if it contains too many missing values
		self.saveLocal() # save ticker_feature_label matrix to local
		
	def loadDict(self):
		f = open('./dat/feature_projection')
		self.wordDict = {}
		for num, line in enumerate(f):
			self.wordDict[line.strip()] = num

	def readFile(self):
		f = open('./dat/' + "_".join(["raw", self.d0, self.d1, self.dtype]))
		res = ""
		for i, line in enumerate(f):
			res += line
		self.lists = re.findall(r"([\w\d\-_%*&/]+:[\w\d\.\-/ ]+)", res) # match key-value pair

	def createFeature(self):
		# turn the feature in a (samples, features) matrix, A is a feature for one sample
		self.feature, A = np.empty([0, 11 * 83]), np.empty([0, 11])
		self.tickerList = [] # use array, since we need the location of each ticker
		self.stock_prices, self.stock_returns = {}, {} # 2-D hash table
		last = -1
		for _, line in enumerate(self.lists): # read key-value pairs
			dat = line.strip().split(":") # split key-value pair
			if dat[0] == "ticker": # everytime update new ticker, clear past array
				ticker = dat[1]
				if np.shape(A) == (83, 11):
					if last != -1 and len(self.stock_prices[last]) > 252 * 3: # data check
						A = A.flatten() # change 2-D matrix to 1-D
						self.tickerList.append(last)
						self.feature = np.vstack([self.feature, A])
				A = np.empty([0, 11])
				self.stock_prices[ticker], self.stock_returns[ticker] = {}, {}
				last = ticker

			if dat[0] in self.wordDict: # if key_ratios are these we need
				numList = np.array(dat[1].split(' ')) # create one row in numpy
				A = np.vstack([A, numList]) # add to A

			m = re.search(r"(\d{4}\-\d{2}\-\d{2})_adjClose", dat[0]) # match stock price
			if m:
				curDate = m.group(1)
				stock_price = float(dat[1])
				self.stock_prices[ticker][curDate] = stock_price
			
		# add the last qualified sample
		if np.shape(A) == (83, 11):
			if last != -1 and len(self.stock_prices[last]) > 252 * 3: # data check
				A = A.flatten() # change 2-D matrix to 1-D
				self.tickerList.append(last)
				self.feature = np.vstack([self.feature, A])
			
	def createLabel(self):
		self.label = np.zeros(len(self.feature), dtype=int)
		if len(self.feature) != len(self.tickerList): 
			sys.exit("feature number doesn't match label information")
		self.deleteList = {}
		for _ in xrange(len(self.tickerList)):
			ticker = self.tickerList[_]
			try:

				price_per = self.stock_prices[ticker]["2016-01-04"] / \
							self.stock_prices[ticker]["2016-09-08"]
				self.label[_] = 1 if price_per > 1 else 0
			except:
				# delete the stock if its price info is not satistifed
				self.deleteList[ticker] = None 
				continue

	def cleanFeatures(self):
		# self.feature = np.nan_to_num(self.feature.astype(np.float))
		# this part should not include ticker info as its 1st col
		self.feature = self.feature.astype(np.float)
		print "Raw feature dimension: ", np.shape(self.feature)
		tag_none_ratio = np.repeat(True, np.shape(self.feature)[1])
		for num in range(np.shape(self.feature)[1]):
			features_j = self.feature[:, num]
			# compute the ratio of missing value in feature_j
			none_ratio = float(len(features_j[np.isnan(features_j)])) / len(features_j)
			if none_ratio > 0.0: tag_none_ratio[num] = False
		self.feature = self.feature[:,tag_none_ratio]
		print "Feature dimension after None deletion: ", np.shape(self.feature)

		# tag true with feature variance is above than threshold, otherwise tag false
		tag_sd = (np.std(self.feature.astype(np.float), axis=0) > 0)
		self.feature = self.feature[:,tag_sd]
		print "Feature dimension after variance check: ", np.shape(self.feature)

		# add ticker to the 1st col, label to the last col of the feature matrix
		self.tickerList = np.array(self.tickerList)
		self.feature = self.feature.transpose()
		self.feature = np.vstack([self.tickerList, self.feature, self.label])
		self.feature = self.feature.transpose()
		print "Feature dimension after col-merge: ", np.shape(self.feature)

		# if the stock don't have coresponding price, delete it.
		tag_price = np.repeat(True, len(self.feature))
		for num in xrange(len(self.feature)):
			if self.feature[num][0] in self.deleteList:
				tag_price[num] = False
		self.feature = self.feature[tag_price]
		print "Feature dimension after price check: ", np.shape(self.feature)

	def interpolationFeatures(self):
		######################### interpolation required ##########################
		## we need interpolation to replace all nan with reasonable value #########
		pass

	def saveLocal(self):
		np.savetxt("./dat/feature_label_" + self.d0 + '_' + self.d1, \
			self.feature, delimiter=',', fmt="%s")



if __name__ == "__main__":
	date_start = "2000-01-01"
	date_end = "2016-12-31"
	date_type = "d"
	s = preprocessing(date_start, date_end, date_type)