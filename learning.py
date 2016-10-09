import numpy as np
from sklearn import svm
from sklearn import metrics

from sklearn.model_selection import GridSearchCV

# from sklearn document
# Proper choice of C and gamma is critical to the SVM performance. 
# use sklearn.model_selection.GridSearchCV with C and gamma 
# spaced exponentially far apart to choose good values.
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV

date_start = "2000-01-01"
date_end = "2016-12-31"
date_type = "d"

feature_path = "./dat/feature_label_" + "_".join([date_start, date_end])
dat = np.loadtxt(feature_path, delimiter=',', dtype='str')

tickers = dat[:,0]

# get rid of the 1st and last cols
features = dat[:, 1:len(dat[0,]) - 50].astype(np.float) 

labels = dat[:,-1].astype(np.int)


print features
print "feature dimension", np.shape(features), len(features[1,]), len(features[:,1])
print labels

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
