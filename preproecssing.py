import os
import re
import collections
import numpy as np

class preprocessing:
	def __init__(self, d1, d2, dtype):
		self.date_start, self.date_end, self.date_type = d1, d2, dtype
		self.loadDict() # load key_ratio list to filter other ratios
		self.readFile() # read file to load features and label information
		self.createFeature() # create feature matrix and get price information
		self.createLabel() # create labels corresponding to features based on price
		self.saveLocal() # save ticker_feature_label matrix to local
		
	def loadDict(self):
		f = open('./dat/feature_projection')
		self.wordDict = {}
		for num, line in enumerate(f):
			self.wordDict[line.strip()] = num

	def readFile(self):
		self.path = './dat/' + "_".join([self.date_start, self.date_end])
		f = open(self.path + '_' + self.date_type)
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
						self.tickerList.append(ticker)
						print np.shape(self.feature), ticker
						self.feature = np.vstack([self.feature, A])
				A = np.empty([0, 11])
				self.stock_prices[ticker], self.stock_returns[ticker] = {}, {}

			if dat[0] in self.wordDict: # if key_ratios are these we need
				numList = np.array(dat[1].split(' ')) # create one row in numpy
				A = np.vstack([A, numList]) # add to A
				A[A == 'None'] = np.nan # change None to the numpy format

			m = re.search(r"adjClose_(\d{4}\-\d{2}\-\d{2})", dat[0]) # match stock price
			if m:
				curDate = m.group(1)
				stock_price = float(dat[1])
				self.stock_prices[ticker][curDate] = stock_price
			last = ticker
			######################### interpolation required ##########################
			## we need interpolation to replace all nan with reasonable value #########

		# delete feature if it contains too many Nones
		cnt_none, length = 0, len(self.feature[1,])
		while cnt_none < length:
			features_j = self.feature[:,cnt_none]
			none_ratio = float(len(features_j[features_j==np.nan])) / len(features_j)
			print none_ratio, len(features_j)
			if none_ratio > 0.10:
				self.feature = np.delete(self.feature, cnt_none, axis=1)
				length -= 1
			else:
				cnt_none += 1
		self.feature = np.nan_to_num(self.feature.astype(np.float))
		# add ticker to the 1st col, label to the last col
		self.tickerList = np.array(self.tickerList)
		self.feature = self.feature.transpose()
		self.feature = np.vstack([self.tickerList, self.feature, self.label])
		self.feature = self.feature.transpose()
		# for ticker in self.stock_prices:
		# 	for num, day in enumerate(sorted(self.stock_prices[ticker])):
		# 		if num > 0:
		# 			self.stock_returns[ticker][day] = self.stock_prices[ticker][day] / lastPrice - 1
		# 		lastPrice = self.stock_prices[ticker][day]

	def createLabel(self):
		# use random number first
		self.label = np.zeros(len(self.feature), dtype=int)
		if len(self.feature) != len(self.tickerList): 
			sys.exit("feature number doesn't match label information")
		for _ in xrange(len(self.tickerList)):
			ticker = self.tickerList[_]
			#od = collections.OrderedDict(sorted(self.stock_prices[ticker].items()))
			#print od.values()
			try:
				print ticker
				price_per = self.stock_prices[ticker]["2016-09-15"] / \
							self.stock_prices[ticker]["2016-09-08"]
				self.label[_] = 1 if price_per > 1 else 0
			except:
				continue

	def saveLocal(self):
		np.savetxt(self.path + "_ticker_feature_label", self.feature, delimiter=',', fmt="%s")






if __name__ == "__main__":
	date_start = "2000-01-01"
	date_end = "2016-12-30"
	date_type = "d"
	s = preprocessing(date_start, date_end, date_type)