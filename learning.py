import numpy as np
import csv
import re
from sklearn import svm
from sklearn import metrics

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from collections import Counter

# from sklearn document
# Proper choice of C and gamma is critical to the SVM performance. 
# use sklearn.model_selection.GridSearchCV with C and gamma 
# spaced exponentially far apart to choose good values.
# http://scikit-learn.org/stable/modules/generated/sklearn.model_\
# selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV


# should update: add another label, one to train model, one to make prediction

class learning:
	def __init__(self, date_start, date_end, date_type):
		self.loadFile()
		self.trainSVM()
		self.predictSVM()

	def loadFile(self):
		typePar = "_".join([date_start, date_end])
		# load feature names
		self.feature_names = np.loadtxt("./dat/selected_feature_" + typePar, dtype='str')
		# load feature_label file, 1st col as tickers, last col as label
		dat = np.loadtxt("./dat/feature_label_" + typePar, delimiter=',', dtype='str')
		self.tickers = dat[:,0]
		self.feature = dat[:, 1:np.shape(dat)[1] - 2].astype(np.float)
		self.label_train = dat[:,-2].astype(np.int)
		self.label_test = dat[:,-1].astype(np.int)


	def selectedTime(self, names, year0, year1, ifTTM): # year1 as int
		tag = np.repeat(True, len(names))
		for num, name in enumerate(names):
			m = re.search("_(\d{4}|TTM)$", name)
			if m:
				date = m.group(1)
				if date == "TTM" and not ifTTM: tag[num] = False
				elif int(date) > year1 or int(date) < year0:
					tag[num] = False
				#if not tag[num]: print "Deleted feature: ", name
		#print "Existing features:", names[tag]
		return tag


	def trainSVM(self):
		self.fout = open('./dat/result_' + "_".join([date_start, date_end]), 'a+') # save to local
		# svm can't handle missing values
		self.feature = np.nan_to_num(self.feature)

		# train model use data from 2006-2014, excluding TTM
		tag = self.selectedTime(self.feature_names, 2006, 2013, 0)
		X, y = self.feature[:, tag], self.label_train

		# sample number
		n_samples = len(self.tickers)
		self.ptLocal(self.fout, "Feature dimension (sample, feature): %d %d\n", np.shape(X))
		# count the number of each label
		label_cnt = Counter(y)
		for _ in label_cnt:
			self.ptLocal(self.fout, "Label %d: number %d\n", (_, label_cnt[_]))
		# cross validation
		k_fold = KFold(3)

		#choose the best C and gamma
		best, scores = 0, {}
		self.ptLocal(self.fout, "\nChoose best parameters\n", ())
		for c in np.logspace(3, 6, 3): # (s, e, n) means: n number starting from 10^s to 10^e
			for gamma in np.logspace(-10, -7, 5):
				score = []
				for k, (train, test) in enumerate(k_fold.split(X, y)):
					classifier = svm.SVC(C=c, gamma=gamma, class_weight={1: 10})
					classifier.fit(X[train], y[train])
					score.append(classifier.score(X[test], y[test]))
				if min(score) > best:
					best = min(score)
					self.ptLocal(self.fout, "C = %.1e, gamma = %.1e, score = %.3f\n", \
						(c, gamma, best))
					self.best_C, self.best_gamma = c, gamma
					self.classifier = classifier
		
	def predictSVM(self):
		# make prediction based on data from 2007-2015, excluding TTM
		tag = self.selectedTime(self.feature_names, 2007, 2014, 0)
		X, expected = self.feature[:, tag], self.label_test
		print np.shape(X)
		predicted = self.classifier.predict(X)
		

		self.ptLocal(self.fout, "Classification report for classifier %s:\n%s\n", \
			(self.classifier, metrics.classification_report(expected, predicted)))
		self.ptLocal(self.fout, "Confusion matrix:\n%s", metrics.confusion_matrix(expected, predicted))

		# use data until Dec, 2014 as input
		# label based on diff between Jul, 2014 and Jun, 2015, select best parameters

		# use this model (based on previous parameters) to train data until Jun.2016
		# predict the best possible stocks which may give us a good Sortino ratio and CVaR
		f_predict = open('./dat/tickers_' + "_".join([date_start, date_end]), 'w')
		tickers_pred = self.tickers[predicted.astype(bool)].tolist()
		f_predict.write(",".join(tickers_pred))
		print "Possible ticker number :", len(tickers_pred)


	def ptLocal(self, fout, content, pars): # pars should be an array
		print(content % (pars))
		fout.write(content % (pars))
		




if __name__ == "__main__":
	date_start = "2000-01-01"
	date_end = "2016-12-31"
	date_type = "d"
	s = learning(date_start, date_end, date_type)
