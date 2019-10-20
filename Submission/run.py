import numpy as np
from implementations import *
from proj1_helpers import *

y,X,ids = load_csv_data("train.csv")

#replace missing values with column median
X = np.where(X == -999., np.nan, X)
col_means = np.nanmedian(X, axis=0)
idxs = np.where(np.isnan(X))
X[idxs] = np.take(col_means, idxs[1])

X = make_feature_split_mass(X)
X = make_feature_momentums(X)
X = make_feature_abs_phi(X)
X = make_feature_ratios(X)
X = make_feature_diff_angles(X)

#feature 6: categorical PRI_jet_num
jet_num_0 = (X[:,22] == 0).astype(int)
jet_num_1 = (X[:,22] == 1).astype(int)
jet_num_2 = (X[:,22] == 2).astype(int)
jet_num_3 = (X[:,22] == 3).astype(int)

X = standardize(X)
#add bias term
X = np.column_stack((np.ones(X.shape[0]), X))
#add categorical after normalization
X = np.column_stack((X, jet_num_0, jet_num_1, jet_num_2, jet_num_3))

