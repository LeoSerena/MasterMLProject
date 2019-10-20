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
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        grad = mse_gradient(y, tx, w)
        w = w - gamma * grad
    loss = compute_loss(y, tx, w)
    return loss, w

def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size = 1):
    
    w = initial_w
    rand_list = np.arange(y.shape[0])
    
    for n in np.arange(max_iters):
        np.random.shuffle(rand_list)
        # randomizing y and tx so we can take the first *batch_size* elements
        y = y[rand_list]
        tx = tx[rand_list]
        
        # compute loss and gradent descent
        grad = mse_gradient(y[:batch_size], tx[:batch_size,:], w)
        
        w = w - gamma * grad
        
    loss = compute_loss(y, tx, w)
    return loss, w

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
    return 1/(1 + np.exp(-z))

def logistic_loss(y, x, w):
    #numerical stability
    eps = 0
    loss = -np.mean(y * np.log(pred(x,w)+eps) + (1-y) * np.log(1-pred(x,w)+eps))
    return loss

def logistic_loss_gradient(y, x, w):
    return np.dot(x.T, (pred(x, w) - y))

def pred(X, w):
  return sigmoid(np.dot(X, w))

def logistic_gradient_descent(y, tx, init_w, max_iter, gamma, batch_size = 1):
    w = init_w
    losses = []
    rand_list = np.arange(y.shape[0])
    for i in range(max_iter):
        np.random.shuffle(rand_list)
        y = y[rand_list]
        tx = tx[rand_list]
            
        grad = logistic_loss_gradient(y[batch_size:], tx[batch_size:], w) / batch_size
        w = w - gamma * grad
        
        loss = logistic_loss(y, tx, w)
        losses.append(loss)
    return losses, w

def classification(x):
    return np.where(x < 1/2, 0, 1)

def reg_logistic_loss(y, tx, w, lambda_):
    return logistic_loss(y, tx, w) - (lambda_ / 2) * w.T @ w

def reg_logistic_gradient(y, tx, w, lambda_):
    return logistic_loss_gradient(y, tx, w) + lambda_* w

def reg_gradient_descent(y, tx, lambda_, init_w, max_iter, gamma, batch_size = 1):
    w = init_w
    rand_list = np.arange(y.shape[0])
    for i in range(max_iter):
        # since often 1, just to save randomization cost
        if batch_size != 1:
            np.random.shuffle(rand_list)
            # randomizing y and tx so we can take the first *batch_size* elements
            y = y[rand_list]
            tx = tx[rand_list]
            
        grad = reg_logistic_gradient(y, tx, w, lambda_)
        w = w - gamma * grad
        
        loss = reg_logistic_loss(y, tx, w, lambda_)
    return loss, w