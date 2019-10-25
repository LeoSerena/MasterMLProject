# -*- coding: utf-8 -*-
import numpy as np

def standardize(x):
    """
    normalization of the data
    
    input
        x, data to be normalized
    output
        x, the normalized data
        mean_x, mean of the data column
        std_x, standard deviation of the data
        """
    mean_x = np.nanmean(x, axis = 0)
    x = x - mean_x
    std_x = np.nanstd(x, axis = 0)
    x = x / std_x
    return x, mean_x, std_x

def compute_loss(y, tx, w):
    """
    computation of the mean square error given the labels, 
    the learned vector and the data
    
    input
        y, the labels
        tx, the data
        w, the learned vector
    output
        the mean square error
    """
    return np.mean((y - tx @ w) ** 2)

def mse_gradient(y, tx, w):
    """
    computation of the gradient of the MSE w.r.t the weights w
    input
        y, the labels
        tx, the data
        w, the vector
    output
        the vectorized gradient for each w
    """
    e = y - np.dot(tx, w)
    grad = -np.dot(tx.T, e)
    return grad

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Gradient descent algorithm
    
    input
        y, the labels
        tx, the trainig data
        initial_w, the first weight vector
        max_iters, the maximum number of GD iterations
        gamma, the learning rate
    output
        w, the last learned weight vector
        loss, the loss at the last iteration
    """
    w = initial_w
    for n_iter in range(max_iters):
        grad = mse_gradient(y, tx, w)
        w = w - gamma * grad / y.shape[0]
    loss = compute_loss(y, tx, w)
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    gradient descent using batch size one
    
    input
        y, the labels
        tx, the trainig data
        initial_w, the first weight vector
        max_iters, the maximum number of GD iterations
        gamma, the learning rate
    output
        w, the last learned weight vector
        loss, the loss at the last iteration
    """
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

def least_squares(y, tx):
    """
    finds least squares analytical solution
    
    input
        y, the labels
        tx, the training data
    output
        w, the learned weight vector
        loss, the loss corresponding to the learned vector
    """
    XT_X = tx.T @ tx
    XT_Y = tx.T @ y
    w = np.linalg.inv(XT_X) @ XT_Y
    loss = compute_loss(y, tx, w)
    return w, loss

def ridge_regression_loss(y, tx, w, lambda_):
    """
    computes the ridge regression loss
    
    input
        y, the labels
        tx, the data
        w, the weights
        lambda_, the penalizing factor
    output
        the ridge loss
    """
    return compute_loss(y, tx, w) + (lambda_ / 2) * w.T @ w

def ridge_regression(y, tx, lambda_):
    """
    finds ridge regression analytical solution
    
    input
        y, the labels
        tx, the training data
        lambda_, the penalizing factor
    output
        w, the learned weight vector
        loss, the loss corresponding to the learned vector
    """ 
   
    XT_X = tx.T @ tx
    XT_Y = tx.T @ y
    w = np.linalg.inv(XT_X + np.eye(tx.shape[1]) * lambda_) @ XT_Y
    loss = ridge_regression_loss(y, tx, w, lambda_)
    return w, loss

def sigmoid(z):
    """
    sigmoid function: 1/(1 + exp(-z))
    
    input
        z, real value
    output
        Sigmoid(z)
    """
    return 1.0/(1 + np.exp(-z))

def logistic_loss(y, x, w):
    """
    computes the logistic loss
    
    input
        y, the labels
        x, the data
        w, the weights
    output
        the computed logistic loss
    """
    eps = 0.0000001
    loss = -np.mean(y * np.log(pred(x,w)+eps) + (1-y) * np.log(1-pred(x,w)+eps))
    return loss

def logistic_loss_gradient(y, x, w):
    """
    computes the logistic loss gradient
    
    input
        y, the labels
        y, the data
        w, the weights
    output
        the computed gradient for each vector
    """
    return np.dot(x.T, (pred(x, w) - y))

def pred(X, w):
    """
    computes the prediction of the data given the weights
    for logistic regression
    
    input
        X, the data to be predicted
        w, the learned weights
    output
        the prediction Sigmoid(X@w)
    """
    return sigmoid(np.dot(X, w))

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    logistic regression using stochastic gradient descent
    
    input
        y, the labels
        tx, the data
        initial_w, the initial weight vector
        max_iters, the number of gradient descent steps
        gamma, the learning rate
    output
        w, the learned weight vector
        loss, the loss corresponding to the last iteration
    """
    w = initial_w
    for i in range(max_iters):
        n = np.random.randint(len(y))
        y_ = y[n]
        tx_ = tx[n]
        grad = logistic_loss_gradient(y_, tx_, w)
        w = w - gamma * grad
        
        
    loss = logistic_loss(y, tx, w)
    return w, loss

def classification(x):
    """
    classifies the data with a 0.5 threshold
    
    input
        x, data to be classified
    output
        data classified
    """
    return np.where(x < 1/2, 0, 1)

def reg_logistic_loss(y, tx, w, lambda_):
    """
    computes the loss for regularized logistic regression
    
    input
        y, the labels
        tx, the data
        w, the weights
        lambda_, the penalizing factor
    output
        the computed loss
    """
    return logistic_loss(y, tx, w) + (lambda_ / 2) * w.T @ w

def reg_logistic_gradient(y, tx, w, lambda_):
    """
    computes the gradient for regularized logistic regression
    
    input
        y, the labels
        tx, the data
        w, the weights
        lambda_, the penalizing factor
    output
        the computed gadient for every weight
    """
    return logistic_loss_gradient(y, tx, w) + lambda_* w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    regularized logistic regression using gradient descent
    
    input
        y, the labels
        tx, the data
        lambda_, the penalizing factor
        initial_w, the initial weights
        max_iters, the number of SGD steps
        lambda_, the penalizing factor
    output
        w, the learned weight
        loss, the loss at last iteration
    """
    w = initial_w
    for i in range(max_iters):
        grad = reg_logistic_gradient(y, tx, w, lambda_)
        w = w - gamma * grad
    loss = reg_logistic_loss(y, tx, w, lambda_)
    return w, loss