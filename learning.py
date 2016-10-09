import numpy as np
from sklearn import svm
from sklearn import metrics

from sklearn.model_selection import GridSearchCV

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
		print "Label number (0, 1): ", sum(self.labels), n_samples - sum(self.labels)

		# Create a classifier: a support vector classifier
		# classifier = svm.SVC(C=10000, gamma=0.0001)
		classifier = svm.SVC(C=10**9, gamma=2.2 * 10**-8) # this gives a great classification in 
		# We learn the digits on the first half of the digits
		classifier.fit(self.features[: n_samples/2], self.labels[: n_samples/2])

		# Now predict the value of the digit on the second half:
		expected = self.labels[n_samples / 2:]
		predicted = classifier.predict(self.features[n_samples/2:])

		print("----------------------------------------------")
		print("Classification report for classifier %s:\n%s\n"
		      % (classifier, metrics.classification_report(expected, predicted)))
		print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


		# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10], 'gamma':[1,0.1,0.001]}
		# svr = svm.SVC()
		# classifier = GridSearchCV(svr, parameters)
		# print classifier.fit(features, labels)

		# print sorted(classifier.cv_results_.keys())


if __name__ == "__main__":
	date_start = "2000-01-01"
	date_end = "2016-12-31"
	date_type = "d"
	s = learning(date_start, date_end, date_type)
