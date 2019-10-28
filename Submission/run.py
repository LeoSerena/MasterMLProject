import numpy as np
from implementations import *
from proj1_helpers import *

#import data and replace undefined values with nan
y,X,ids = load_csv_data("train.csv")
X = np.where(X == -999., np.nan, X)

#select indices correspond to different values of 
cnd_1 = X[:,22] == 0
cnd_2 = X[:,22] == 1
cnd_3 = X[:,22] == 2
cnd_4 = X[:,22] == 3

#create seperate data set using the indices
y_1 = y[cnd_1]
X_1 = X[cnd_1]
y_2 = y[cnd_2]
X_2 = X[cnd_2]
y_3 = y[cnd_3]
X_3 = X[cnd_3]
y_4 = y[cnd_4]
X_4 = X[cnd_4]

#preprocess each data set
X_1 = preproc(X_1)
X_2 = preproc(X_2)
X_3 = preproc(X_3)
X_4 = preproc(X_4)

np.random.seed(1)
in_dim = X_1.shape[1]
out_dim = 1

print("Start training")

mlp_1 = MLP(gamma = 0.01, dimensions = [in_dim,60,60,30,out_dim], activations = ['relu','relu','relu','sigmoid'],
          weight_decay = 0.001)
mlp_2 = MLP(gamma = 0.001, dimensions = [in_dim,60,60,60,30,out_dim], activations = ['relu','relu','relu','relu','sigmoid'],
          weight_decay = 0.003)
mlp_3 = MLP(gamma = 0.01, dimensions = [in_dim,60,60,30,out_dim], activations = ['relu','relu','relu','sigmoid'],
          weight_decay = 0.003)
mlp_4 = MLP(gamma = 0.01, dimensions = [in_dim,60,30,out_dim], activations = ['relu','relu','sigmoid'],
          weight_decay = 0.005)
mlp_1.train(X_1,y_1,max_iter = 3500000,decay_rate = 5,decay_iteration = 1500000,decay = True)
mlp_2.train(X_2,y_2,max_iter = 3500000,decay_rate = 10,decay_iteration = 1500000,decay = True)
mlp_3.train(X_3,y_3,max_iter = 3500000,decay_rate = 5,decay_iteration = 1500000,decay = True)
mlp_4.train(X_4,y_4,max_iter = 3500000,decay_rate = 5,decay_iteration = 1500000,decay = True)

#train accuracy
y_1_pred = mlp_1.predict(X_1)
acc1 = 1-np.sum(np.abs(y_1_pred - y_1)) / X_1.shape[0]
print(acc1)
y_2_pred = mlp_2.predict(X_2)
acc2 = 1-np.sum(np.abs(y_2_pred - y_2)) / X_2.shape[0]
print(acc2)
y_3_pred = mlp_3.predict(X_3)
acc3 = 1-np.sum(np.abs(y_3_pred - y_3)) / X_3.shape[0]
print(acc3)
y_4_pred = mlp_4.predict(X_4)
acc4 = 1-np.sum(np.abs(y_4_pred - y_4)) / X_4.shape[0]
print(acc4)

#prepare submission data in the same way as the training data
_,X_sub,ids = load_csv_data("test.csv")
X_sub = np.where(X_sub == -999., np.nan, X_sub)

cnd_1 = X_sub[:,22] == 0
cnd_2 = X_sub[:,22] == 1
cnd_3 = X_sub[:,22] == 2
cnd_4 = X_sub[:,22] == 3

X_sub_1 = X_sub[cnd_1]
X_sub_2 = X_sub[cnd_2]
X_sub_3 = X_sub[cnd_3]
X_sub_4 = X_sub[cnd_4]

X_sub_1 = preproc(X_sub_1)
X_sub_2 = preproc(X_sub_2)
X_sub_3 = preproc(X_sub_3)
X_sub_4 = preproc(X_sub_4)

#generate the prediction for each network's examples
sub_1_pred = mlp_1.predict(X_sub_1)
sub_2_pred = mlp_2.predict(X_sub_2)
sub_3_pred = mlp_3.predict(X_sub_3)
sub_4_pred = mlp_4.predict(X_sub_4)

#map prediction to {-1,1}
sub_1_pred = sub_1_pred*2 -1
sub_2_pred = sub_2_pred*2 -1
sub_3_pred = sub_3_pred*2 -1
sub_4_pred = sub_4_pred*2 -1
#create final submission array
sub_pred = np.zeros(X_sub.shape[0])
sub_pred[cnd_1] = sub_1_pred
sub_pred[cnd_2] = sub_2_pred
sub_pred[cnd_3] = sub_3_pred
sub_pred[cnd_4] = sub_4_pred

create_csv_submission(ids, sub_pred, "nn_4_split_final_submission_script.csv")
