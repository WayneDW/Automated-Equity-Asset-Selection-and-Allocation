import numpy as np
features_j = np.array(['43.0','nan', 'nan', '34.3'])
features_j = np.array(['nan','nan', 'nan', 'nan'])
#features_j = np.array([1,2,3,4])

features_j = features_j.astype(float)

print len(features_j[np.isnan(features_j)])