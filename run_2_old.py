import numpy as np
from implementations import *
from proj1_helpers import *

y,X,ids = load_csv_data("train.csv")

X = np.where(X == -999., np.nan, X)
y_1 = y[~np.isnan(X).any(axis=1)]
X_1 = X[~np.isnan(X).any(axis=1)]
y_2 = y[np.isnan(X).any(axis=1)]
X_2 = X[np.isnan(X).any(axis=1)]

X_1 = preproc(X_1)
X_2 = preproc(X_2)

cutoff_1 = int(0.8*((X_1.shape)[0]))
cutoff_2 = int(0.8*((X_2.shape)[0]))
X_1_train = X_1#[:cutoff_1]
y_1_train = y_1#[:cutoff_1]
X_1_test = X_1[cutoff_1:]
y_1_test = y_1[cutoff_1:]
X_2_train = X_2#[:cutoff_2]
y_2_train = y_2#[:cutoff_2]
X_2_test = X_2[cutoff_2:]
y_2_test = y_2[cutoff_2:]

np.random.seed(1)
in_dim = X_1.shape[1]
n_h1 = 100
n_h2 = 100
n_h3 = 100
n_h4 = 100
n_h5 = 100
n_h6 = 100
n_h7 = 100
out_dim = 1
dimensions = [in_dim, n_h1,n_h2,n_h3,n_h4,n_h5,n_h6,n_h7,out_dim]
activations = ['relu','relu','relu','relu','relu','relu','relu','sigmoid']
gamma = 0.001
weight_decay = 0.001
mlp_1 = MLP(gamma = 0.01, dimensions = [in_dim,60,60,30,out_dim], activations = ['relu','relu','relu','sigmoid'],
          weight_decay = weight_decay)
mlp_2 = MLP(gamma = gamma, dimensions = dimensions, activations = activations,
          weight_decay = weight_decay)
mlp_1.train(X_1_train,y_1_train,max_iter = 3500000,decay_rate = 5,decay_iteration = 1500000,decay = True)
mlp_2.train(X_2_train,y_2_train,max_iter = 3000000,decay_rate = 10,decay_iteration = 1000000,decay = True)

_,X_sub,ids = load_csv_data("test.csv")
X_sub = np.where(X_sub == -999., np.nan, X_sub)
no_nan_idxs = ~np.isnan(X_sub).any(axis=1)
nan_idxs = np.isnan(X_sub).any(axis=1)

X_sub_1 = X_sub[~np.isnan(X_sub).any(axis=1)]
X_sub_2 = X_sub[np.isnan(X_sub).any(axis=1)]

X_sub_1 = preproc(X_sub_1)
X_sub_2 = preproc(X_sub_2)

sub_1_pred = mlp_1.predict(X_sub_1)
sub_2_pred = mlp_2.predict(X_sub_2)
sub_1_pred = sub_1_pred*2 -1
sub_2_pred = sub_2_pred*2 -1
sub_pred = np.zeros(X_sub.shape[0])
sub_pred[no_nan_idxs] = sub_1_pred
sub_pred[nan_idxs] = sub_2_pred

create_csv_submission(ids, sub_pred, "nn_split_submission.csv")