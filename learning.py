import numpy as np
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

class learning:
	def __init__(self, date_start, date_end, date_type):
		# load feature_label file
		feature_path = "./dat/feature_label_" + "_".join([date_start, date_end])
		dat = np.loadtxt(feature_path, delimiter=',', dtype='str')

		self.tickers = dat[:,0]
		# get rid of the 1st and last cols
		self.features, self.labels = dat[:, 1:].astype(np.float), dat[:,-1].astype(np.int)
		self.svm()

	def svm(self):
		self.features = np.nan_to_num(self.features) # svm can't handle missing values
		n_samples = len(self.tickers)
		print "Feature dimension (sample, feature): ", np.shape(self.features)
		label_cnt = Counter(self.labels)
		for _ in label_cnt:
			print("Label %d: number %d" % (_, label_cnt[_]))

		#choose the best C and gamma
		scores = {}
		for c in np.logspace(3, 6, 10):
			for gamma in np.logspace(-11, -8, 20):
				classifier = svm.SVC(C=c, gamma=gamma, class_weight={1: 10})
				classifier.fit(self.features[: n_samples/2], self.labels[: n_samples/2])
				score = classifier.score(self.features[n_samples/2:], self.labels[n_samples/2:])
				print("C = %.1e, gamma = %.1e, score = %.3f" % (c, gamma, score))
				scores[score] = (c, gamma)
		

		best_C, best_gamma = scores[sorted(scores.keys(), reverse=True)[0]]
		print("\nPick best parameters c=%e, gamma=%e"% (best_C, best_gamma))
		# best_C = 3 * 10 ** 5
		# best_gamma = 3 * 10 ** -10
		classifier = svm.SVC(C=best_C, gamma=best_gamma, class_weight={1: 10})
		classifier.fit(self.features[: n_samples/2], self.labels[: n_samples/2])

		# Now predict the value of the digit on the second half:
		expected = self.labels[n_samples / 2:]
		predicted = classifier.predict(self.features[n_samples/2:])

		print("----------------------------------------------")
		print("Classification report for classifier %s:\n%s\n"
		      % (classifier, metrics.classification_report(expected, predicted)))
		print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))



if __name__ == "__main__":
	date_start = "2000-01-01"
	date_end = "2016-12-31"
	date_type = "d"
	s = learning(date_start, date_end, date_type)
