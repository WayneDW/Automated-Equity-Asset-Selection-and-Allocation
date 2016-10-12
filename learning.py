import numpy as np
import csv
import re
from sklearn import svm
from sklearn import metrics
from datetime import datetime

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import scale
from sklearn.neural_network import MLPClassifier
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
		self.dataProcressing()
		#self.SVM()
		self.neuralNet()

	def loadFile(self):
		typePar = "_".join([date_start, date_end])
		# load feature names
		self.feature_names = np.loadtxt("./dat/selected_feature_" + typePar, dtype='str')
		# load feature_label file, 1st col as tickers, last col as label
		dat = np.loadtxt("./dat/feature_label_" + typePar, delimiter=',', dtype='str')
		self.tickers = dat[:,0]
		self.feature = dat[:, 1:np.shape(dat)[1] - 5].astype(np.float)
		self.label_train = {}
		for i in range(4):
			self.label_train[i] = dat[:,-5 + i].astype(np.int)
		self.label_test = dat[:,-1].astype(np.int)

		imp = Imputer(missing_values='NaN', strategy='median', axis=0)
		imp.fit(self.feature)
		self.feature = imp.transform(self.feature)

		self.fout = open('./dat/result_' + "_".join([date_start, date_end]), 'a+') # save to local
		self.fticker = open('./dat/tickers_' + "_".join([date_start, date_end]), 'w')
		curtime = str(datetime.now())
		self.ptLocal(self.fout, "Today: %s",curtime[:len(curtime) - 10])

	def ptLocal(self, fout, content, pars): # pars should be an array
		print(content % (pars))
		content += "\n"
		fout.write(content % (pars))

	def dataProcressing(self):
		# train model use data from 2006-2011, 2007-12, 2008-13, 2009-14
		for i in range(4):
			tag = self.selectedTime(self.feature_names, 2006 + i, 2011 + i, 0)
			if i == 0: self.X, self.y = self.feature[:, tag], self.label_train[i]
			else:
				self.X = np.vstack([self.X, self.feature[:, tag]])
				self.y = np.vstack([self.y, self.label_train[i]])
		self.y = self.y.flatten()
		print np.shape(self.X), np.shape(self.y)
		# sample number
		n_samples = len(self.tickers)
		self.ptLocal(self.fout, "Feature dimension (sample, training feature): %d %d\n", \
			np.shape(self.X))
		# count the number of each label
		label_cnt = Counter(self.y)
		for _ in label_cnt:
			self.ptLocal(self.fout, "Label %d: number %d", (_, label_cnt[_]))

		# make prediction based on data from 2010-2015, excluding TTM
		tag = self.selectedTime(self.feature_names, 2010, 2015, 0)
		self.X_test, self.expected = self.feature[:, tag], self.label_test
		# cross validation
		self.K = 3
		self.k_fold = KFold(self.K)
		


	def selectedTime(self, names, year0, year1, ifTTM): # year1 as int
		tag = np.repeat(True, len(names))
		for num, name in enumerate(names):
			m = re.search("_(\d{4}|TTM)$", name)
			if m:
				date = m.group(1)
				if date == "TTM" and not ifTTM: tag[num] = False
				elif int(date) > year1 or int(date) < year0:
					tag[num] = False
		return tag


	def SVM(self):		
		#choose the best C and gamma
		best, scores = 0, {}
		self.ptLocal(self.fout, "\nChoose best parameters\n", ())
		for c in np.logspace(1, 8, 10): # (s, e, n) means: n number starting from 10^s to 10^e
			for gamma in np.logspace(-7, -2, 10):
				score = []
				for k, (train, test) in enumerate(self.k_fold.split(self.X, self.y)):
					clf = svm.SVC(C=c, gamma=gamma)
					clf.fit(self.X[train], self.y[train])
					score.append(clf.score(self.X[test], self.y[test]))
				if min(score) > best:
					best = min(score)
					self.ptLocal(self.fout, "C = %.1e, gamma = %.1e, score = %.3f\n", \
						(c, gamma, best))
		
		print np.shape(self.X_test)
		predicted = clf.predict(self.X_test)
		
		self.ptLocal(self.fout, "Classification report for classifier %s:\n%s\n", \
			(clf, metrics.classification_report(self.expected, predicted)))
		self.ptLocal(self.fout, "Confusion matrix:\n%s", \
			metrics.confusion_matrix(self.expected, predicted))

		
		tickers_pred = self.tickers[predicted.astype(bool)].tolist()
		self.fticker.write(",".join(tickers_pred))
		print "Possible ticker number:", len(tickers_pred)
		print "Random pick successful ratio: ", \
			round(float(sum(self.expected)) / len(self.expected), 3)

	def neuralNet(self):
		def combine(n, k): # generate Combination 
			if k == 1:
				return [[i] for i in range(1, n+1)]
			if n == k:
				return [[i for i in range(1, n+1)]]
			return [i for i in combine(n-1, k)] + [i + [n] for i in combine(n-1,k-1)]
		
		best = 0
		for layer_n in range(1, 4): # number of layers
			layers = combine(20, layer_n) # combination of layers
			for layer in layers:
				if layer[0] == 1: continue # hidden layer should have more than 1 node
				layer = sorted(layer, reverse=True)
				layer.insert(layer_n, 2)
				score = [0] * self.K
				for k, (train, test) in enumerate(self.k_fold.split(self.X, self.y)):
					clf = MLPClassifier(solver='lbfgs', alpha=1e-5,\
						hidden_layer_sizes=layer, random_state=1)
					clf.fit(self.X[train], self.y[train])
					#score[k] = clf.score(self.X[test], self.y[test]) # if we care the whole
					predicted = clf.predict(self.X[test])
					score_mat = metrics.confusion_matrix(self.y[test], predicted)

					if score_mat[1, 1] == 0: continue # only care label-1 performance
					recall = float(score_mat[1, 1]) / sum(score_mat[1])
					precision = float(score_mat[1, 1]) / sum(score_mat[:, 1])
					score[k] = 1 / ((1 / recall + 1 / precision) / 2) # f1 score
				#if min(score) <= best: continue # only update if we have better result
				best, idx = min(score), score.index(min(score))
				self.ptLocal(self.fout, "\nLayers: %s", ("-".join(np.array(layer, dtype=str))))
				self.ptLocal(self.fout, "Confusion matrix:\n%s", score_mat)
				self.ptLocal(self.fout, "Label 1 expected f1 score: %.3f\n", best)
				clf.fit(self.X, self.y)
				predicted = clf.predict(self.X_test)
				self.ptLocal(self.fout, "Classification report for classifier %s:\n%s", \
					(clf, metrics.classification_report(self.expected, predicted)))
				self.ptLocal(self.fout, "Confusion matrix:\n%s", \
					metrics.confusion_matrix(self.expected, predicted))
				tickers_pred = self.tickers[predicted.astype(bool)].tolist()
				self.fticker.write(",".join(tickers_pred))
				self.ptLocal(self.fout, "Possible ticker number: %s", len(tickers_pred))
				self.ptLocal(self.fout, "Random pick successful rate: %.3f\n",\
				 round(float(sum(self.expected)) / len(self.expected), 3))


if __name__ == "__main__":
	date_start = "2000-01-01"
	date_end = "2016-12-31"
	date_type = "d"
	s = learning(date_start, date_end, date_type)
