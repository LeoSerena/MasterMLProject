# -*- coding: utf-8 -*-
import numpy as np

def standardize(x):
    mean_x = np.nanmean(x, axis = 0)
    x = x - mean_x
    std_x = np.nanstd(x, axis = 0)
    x = x / std_x
    return x, mean_x, std_x


# feature 1 : split der_mass
def make_feature_split_mass(X):
    X_gt_mmc = np.array(X[:,0], copy=True)
    X_gt_mmc[X_gt_mmc <= 140] = 140
    X[:,0][X[:,0] > 140] = 140
    X = np.column_stack((X, X_gt_mmc))
    return X

# feature 2 : add momentums
def make_feature_momentums(X):
    #add tau momentums
    tau_px = X[:,13]*np.cos(X[:,15])
    tau_py = X[:,13]*np.sin(X[:,15])
    tau_pz = X[:,13]*np.sinh(X[:,14])
    X = np.column_stack((X, tau_px,tau_py,tau_pz))
    #add lep momentums
    lep_px = X[:,16]*np.cos(X[:,18])
    lep_py = X[:,16]*np.cos(X[:,18])
    lep_pz = X[:,16]*np.cos(X[:,17])
    X = np.column_stack((X, lep_px,lep_py,lep_pz))
    #add leading jet momentums
    jet_px = X[:,23]*np.cos(X[:,25])
    jet_py = X[:,23]*np.cos(X[:,25])
    jet_pz = X[:,23]*np.cos(X[:,24])
    X = np.column_stack((X, jet_px,jet_py,jet_pz))
    #add subleading jet momentums
    subjet_px = X[:,26]*np.cos(X[:,28])
    subjet_py = X[:,26]*np.cos(X[:,28])
    subjet_pz = X[:,26]*np.cos(X[:,27])
    X = np.column_stack((X, subjet_px,subjet_py,subjet_pz))
    return X

# #feature 3: abs phi angles
def make_feature_abs_phi(X):
    #der_met_phi_centrality
    X[:,11] = np.abs(X[:,11])
    #tau phi
    X[:,15] = np.abs(X[:,15])
    #lep phi
    X[:,18] = np.abs(X[:,18])
    #met phi
    X[:,20] = np.abs(X[:,20])
    #lead jet phi
    X[:,24] = np.abs(X[:,24])
    #sublead jet phi
    X[:,27] = np.abs(X[:,27])
    return X

# #feature 4: ratios
def make_feature_ratios(X):
    tau_lep_ratio = X[:,13]/X[:,16]
    met_tot_ratio = X[:,19]/X[:,21]
    X = np.column_stack((X, tau_lep_ratio,met_tot_ratio))
    return X

# #feature 5: jets_diff_angle
def make_feature_diff_angle(X):
    jets_diff_angle = np.cos(X[:,24]-X[:,27])
    X = np.column_stack((X, jets_diff_angle))
    return X

def compute_loss(y, tx, w, mse = True):
    N = y.shape[0]
    if mse:
        e = y - tx @ w
        loss = 1/(2 * N) * e.T @ e
    else:
        loss = np.mean(np.abs(y - tx @ w))
    return loss

def mse_gradient(y, tx, w):
    e = y - tx @ w
    grad = -(1/y.shape[0]) * tx.T @ e
    return grad

def mae_gradient(y, tx, w):
    e = y - tx @ w
    e = np.where(e < 0, 1, -1)
    e = np.vstack((e,e)).T * tx
    return np.mean(e, axis = 0)

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for i in range(max_iters):
        grad = mse_gradient(y, tx, w)
        w = w - gamma * grad
    loss = compute_loss(y, tx, w)
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    N = tx.shape[0]
    for i in range(max_iters):
        idx = np.random.randint(N)
        y_b = y[idx]
        tx_b = tx[idx]
        grad = mse_gradient(y_b, tx_b, w)
        w = w - gamma * grad
        
    loss = compute_loss(y, tx, w)
    return w, loss

def least_square_loss(tx, y, w):
    e = y - tx @ w.T
    return (1/y.shape[0]) * e @ e.T

def least_squares(tx, y):
    XT_X = tx.T @ tx
    XT_Y = tx.T @ y
    w = np.linalg.inv(XT_X) @ XT_Y
    loss = least_square_loss(tx, y, w)
    return w, loss

def ridge_regression_loss(tx, y, w, lambda_):
    return least_square_loss(tx, y, w) + lambda_ * w.T @ w

def ridge_regression(tx, y, lambda_):
    XT_X = tx.T @ tx
    XT_Y = tx.T @ y
    w = np.linalg.inv(XT_X + np.eye(tx.shape[1]) * lambda_) @ XT_Y
    loss = ridge_regression_loss(tx, y, w, lambda_)
    return w, loss

def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

def logistic_loss(y, x, w):
    #numerical stability
    eps = 0
    loss = -np.mean(y * np.log(pred(x,w)+eps) + (1-y) * np.log(1-pred(x,w)+eps))
    return loss

def logistic_gradient(y, x, w):
    return np.dot(x.T,(pred(x, w) - y))

def pred(x, w):
    return sigmoid(np.dot(x,w))

#LR with GD
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for i in range(max_iters):
        grad = (1 / y.shape[0]) * logistic_gradient(y, tx, w)
        w = w - gamma * grad
    loss = logistic_loss(y, tx, w)
    return w,loss

def classification(x):
    return np.where(x < 1/2, 0, 1)

def reg_logistic_loss(y, tx, w, lambda_):
    return logistic_loss(y, tx, w) + (lambda_ / 2) * w.T @ w

def reg_logistic_gradient(y, tx, w, lambda_):
    return logistic_gradient(y, tx, w) + lambda_* w

#LR with L2 reg and SGD    
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    for i in range(max_iters):
        if (i % 100 == 0) and (i != 0):
            loss = reg_logistic_loss(y, tx, w,lambda_)
            print("Iteration:{}, loss : {}".format(i,loss))
        grad = reg_logistic_gradient(y, tx, w, lambda_)
        w = w - gamma * grad        
    loss = reg_logistic_loss(y, tx, w, lambda_)
    return w,loss