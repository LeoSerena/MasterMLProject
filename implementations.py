# -*- coding: utf-8 -*-
import numpy as np

# standardization
def standardize(x):
    mean_x = np.nanmean(x, axis = 0)
    x = x - mean_x
    std_x = np.nanstd(x, axis = 0)
    x = x / std_x
    return x, mean_x, std_x

# MSE loss
def compute_loss(y, tx, w):
    return np.mean((y - tx @ w) ** 2)

# gradients
def mse_gradient(y, tx, w):
    e = y - np.dot(tx, w)
    grad = -np.dot(tx.T, e)
    return grad

# least squaresusing full batch gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        grad = mse_gradient(y, tx, w)
        w = w - gamma * grad / y.shape[0]
    loss = compute_loss(y, tx, w)
    return w, loss

# least squares using stochastic with batch size one
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    
    for n in np.arange(max_iters):
        k = np.random.randint(len(y))
        y_ = y[k]
        x_ = tx[k,:]
        
        # compute gradent descent
        grad = mse_gradient(y_, x_, w)
        
        w = w - gamma * grad
        
    loss = compute_loss(y, tx, w)
    return w, loss

# finding analytical solution for least squares
def least_squares(tx, y):
    XT_X = tx.T @ tx
    XT_Y = tx.T @ y
    w = np.linalg.inv(XT_X) @ XT_Y
    loss = compute_loss(y, tx, w)
    return w, loss

# ridge regression loss
def ridge_regression_loss(y, tx, w, lambda_):
    return compute_loss(y, tx, w) + (lambda_ / 2) * w.T @ w

# finding analytical solution for ridge regression
def ridge_regression(tx, y, lambda_):
    XT_X = tx.T @ tx
    XT_Y = tx.T @ y
    w = np.linalg.inv(XT_X + np.eye(tx.shape[1]) * lambda_) @ XT_Y
    loss = ridge_regression_loss(y, tx, w, lambda_)
    return w, loss

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def logistic_loss(y, x, w):
    #numerical stability
    eps = 0.0001
    loss = -np.mean(y * np.log(pred(x,w)+eps) + (1-y) * np.log(1-pred(x,w)+eps))
    return loss

def logistic_loss_gradient(y, x, w):
    return np.dot(x.T, (pred(x, w) - y))

def pred(X, w):
  return sigmoid(np.dot(X, w))

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    losses = []
    for i in range(max_iters):
        n = np.random.randint(len(y))
        y_ = y[n]
        tx_ = tx[n]
            
        grad = logistic_loss_gradient(y, tx, w)
        w = w - gamma * grad
        
        
    loss = logistic_loss(y, tx, w)
    return w, loss

def classification(x):
    return np.where(x < 1/2, 0, 1)

def reg_logistic_loss(y, tx, w, lambda_):
    return logistic_loss(y, tx, w) - (lambda_ / 2) * w.T @ w

def reg_logistic_gradient(y, tx, w, lambda_):
    return logistic_loss_gradient(y, tx, w) + lambda_* w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    rand_list = np.arange(y.shape[0])
    for i in range(max_iters):
        n = np.random.randint(len(y))
        y_ = y[n]
        tx_ = tx[n]
            
        grad = reg_logistic_gradient(y_, tx_, w, lambda_)
        w = w - gamma * grad
        
    loss = reg_logistic_loss(y, tx, w, lambda_)
    return w, loss