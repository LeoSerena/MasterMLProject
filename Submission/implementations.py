# -*- coding: utf-8 -*-
import numpy as np

def standardize(x):
    mean_x = np.nanmean(x, axis = 0)
    x = x - mean_x
    std_x = np.nanstd(x, axis = 0)
    x = x / std_x
    return x, mean_x, std_x

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