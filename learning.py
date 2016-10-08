import numpy as np
from sklearn import svm
from sklearn import metrics

from preprocessing import preprocessing
from sklearn.model_selection import GridSearchCV

# from sklearn document
# Proper choice of C and gamma is critical to the SVM performance. 
# use sklearn.model_selection.GridSearchCV with C and gamma 
# spaced exponentially far apart to choose good values.
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV

date_start = "2000-01-01"
date_end = "2016-12-30"
date_type = "d"

feature_path = "./dat/" + "_".join([date_start, date_end, "ticker_feature_label"])
dat = np.loadtxt(feature_path, delimiter=',', dtype='str')

tickers = dat[:,0]
# get rid of the 1st and last cols
features = dat[:, 1:len(dat[0,]) - 1].astype(np.float) 
labels = dat[:,-1].astype(np.int)

i, length = 0, len(features[1,])
while i < length:
	features_j = features[:,i]
	none_ratio = float(len(features_j[features_j==0])) / len(features_j)
	if none_ratio > 0.01:
		features = np.delete(features, i, axis=1)
		length -= 1
	else:
		i += 1
print len(features[1,])
print features

n_samples = len(tickers)
#print "sample number", n_samples

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.1)

# We learn the digits on the first half of the digits
classifier.fit(features[: n_samples/2], labels[: n_samples/2])

# Now predict the value of the digit on the second half:
expected = labels[n_samples / 2:]
predicted = classifier.predict(features[n_samples/2:])
print("----------------------------------------------")
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


print np.mean(labels)

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10], 'gamma':[1,0.1,0.001]}
svr = svm.SVC()
classifier = GridSearchCV(svr, parameters)
print classifier.fit(features, labels)

print sorted(classifier.cv_results_.keys())
