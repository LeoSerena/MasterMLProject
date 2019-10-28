# -*- coding: utf-8 -*-
import numpy as np

def compute_loss(y, tx, w, mse = True):
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
    N = y.shape[0]
    if mse:
        e = y - tx @ w
        loss = 1/(2 * N) * e.T @ e
    else:
        loss = np.mean(np.abs(y - tx @ w))
    return loss

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
    e = y - tx @ w
    grad = -(1/y.shape[0]) * tx.T @ e
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
    for i in range(max_iters):
        grad = mse_gradient(y, tx, w)
        w = w - gamma * grad
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
    e = y - tx @ w.T
    return (1/y.shape[0]) * e @ e.T

def least_squares(tx, y):
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
    loss = least_square_loss(tx, y, w)
    return w, loss

def ridge_regression_loss(tx, y, w, lambda_):
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
    return least_square_loss(tx, y, w) + lambda_ * w.T @ w

def ridge_regression(tx, y, lambda_):
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
    loss = ridge_regression_loss(tx, y, w, lambda_)
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
    eps = 0
    loss = -np.mean(y * np.log(pred(x,w)+eps) + (1-y) * np.log(1-pred(x,w)+eps))
    return loss

def logistic_gradient(y, x, w):
    """
    computes the logistic loss gradient
    
    input
        y, the labels
        y, the data
        w, the weights
    output
        the computed gradient for each vector
    """
    return np.dot(x.T,(pred(x, w) - y))

def pred(x, w):
    """
    computes the prediction of the data given the weights
    for logistic regression
    
    input
        X, the data to be predicted
        w, the learned weights
    output
        the prediction Sigmoid(X@w)
    """
    return sigmoid(np.dot(x,w))

#LR with GD
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
        grad = (1 / y.shape[0]) * logistic_gradient(y, tx, w)
        w = w - gamma * grad
    loss = logistic_loss(y, tx, w)
    return w,loss

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
    return logistic_gradient(y, tx, w) + lambda_* w

#LR with L2 reg and SGD    
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
        if (i % 100 == 0) and (i != 0):
            loss = reg_logistic_loss(y, tx, w,lambda_)
            print("Iteration:{}, loss : {}".format(i,loss))
        grad = reg_logistic_gradient(y, tx, w, lambda_)
        w = w - gamma * grad        
    loss = reg_logistic_loss(y, tx, w, lambda_)
    return w,loss

#split DER_mass_mcc in two features, and floor or caps each by 140. See report for more details.
def make_feature_mass(X):
    X_gt_mmc = np.array(X[:,0], copy=True)
    X_gt_mmc[X_gt_mmc <= 140] = 140
    X[:,0][X[:,0] > 140] = 140
    X = np.column_stack((X, X_gt_mmc))
    return X

#add momentum and modulues values and estimated total invariant mass. See report for more details.
def make_feature_moms_inv_mass(X):
    #tau momentum and modulus
    tau_px = X[:,13]*np.cos(X[:,15])
    tau_py = X[:,13]*np.sin(X[:,15])
    tau_pz = X[:,13]*np.sinh(X[:,14])
    tau_mod = X[:,13]*np.cosh(X[:,14])
    X = np.column_stack((X, tau_px,tau_py,tau_pz,tau_mod))
    #lep momentum and modulus
    lep_px = X[:,16]*np.cos(X[:,18])
    lep_py = X[:,16]*np.sin(X[:,18])
    lep_pz = X[:,16]*np.sinh(X[:,17])
    lep_mod = X[:,16]*np.cosh(X[:,17])
    X = np.column_stack((X, lep_px,lep_py,lep_pz,lep_mod))
    #leading jet momentum and modulus
    jet_px = X[:,23]*np.cos(X[:,25])
    jet_py = X[:,23]*np.sin(X[:,25])
    jet_pz = X[:,23]*np.sinh(X[:,24])
    jet_mod = X[:,23]*np.cosh(X[:,24])
    X = np.column_stack((X, jet_px,jet_py,jet_pz,jet_mod))
    #subleading jet momentum and modulus
    subjet_px = X[:,26]*np.cos(X[:,28])
    subjet_py = X[:,26]*np.sin(X[:,28])
    subjet_pz = X[:,26]*np.sinh(X[:,27])
    subjet_mod = X[:,26]*np.cosh(X[:,27])
    X = np.column_stack((X, subjet_px,subjet_py,subjet_pz,subjet_mod))
    #add total invariant mass
    term_1 = np.sqrt(tau_px**2 + tau_py**2 + tau_pz**2) + np.sqrt(lep_px**2 + lep_py**2 + lep_pz**2) \
    + np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2) + np.sqrt(subjet_px**2 + subjet_py**2 + subjet_pz**2)
    term_2 = (tau_px + lep_px + jet_px + subjet_px)**2 + (tau_py + lep_py + jet_py + subjet_py)**2 \
            + (tau_pz + lep_pz + jet_pz + subjet_pz)**2
    inv_mass = np.sqrt(term_1**2 - term_2)
    X = np.column_stack((X, inv_mass))
    return X

#add inverse of log of features + 1 for stability.
def make_feature_inv_log(X):
    #     feature 4: inverse log
    log_cols = [0,1,2,3,4,5,7,8,9,10,12,13,16,19,21,23,26]
    X_log_cols = np.log(1 / (1 + X[:, log_cols]))
    X = np.hstack((X, X_log_cols))
    return X

#add 2 ratios
def make_feature_ratios(X):
    # #tau_lep_ratio = PRI_tau_pt/PRI_lep_pt
    tau_lep_ratio = X[:,13]/X[:,16]
    # #met_tot_ratio = PRI_met/PRI_met_sumet
    met_tot_ratio = X[:,19]/X[:,21]
    X = np.column_stack((X, tau_lep_ratio,met_tot_ratio))
    return X
    

def preproc(X):
    #if column is all nan, set to 0, otherwise replace nans with column median
    for i in range(X.shape[1]):
        if (np.isnan(X[:,i]).all()):
            X[:,i] = 0
        else:
            col_means = np.nanmedian(X[:,i])
            idxs = np.where(np.isnan(X[:,i]))
            X[idxs,i] = col_means

    #generate features
    X = make_feature_mass(X)
    X = make_feature_moms_inv_mass(X)
    X = make_feature_inv_log(X)
    X = make_feature_ratios(X)
    
    X = normalize(X)
    
    return X


#normalize columns; set to 0 if var = 0; add bias
def normalize(X):
    """
    normalizes the data or sets the column to 0 if the standard deviation is 0.
    
    input
       x, data to be normalized
    output
       x, the normalized data
    """
    #set to 0 if column has no variance
    for i in range(X.shape[1]):
        if (X[:,i].std() == 0):
            X[:,i] = 0
        else:
            X[:,i] = (X[:,i] - X[:,i].mean()) / X[:,i].std()
    return np.column_stack((np.ones(X.shape[0]), X))

class MLP:
    """
    Creates a fully connected neural network with given layer size and activation functions. The loss function is
    binary cross-entropy with L2 regularization (weight decay). It supports stochastic gradient descent with batch
    size of 1.
    
    input
        gamma, the learning rate used for gradient descent
        dimensions, the dimensions of the layers (including input and output)
        activations, the activation functions used for each layer. Either 'relu', 'sigmoid' or 'linear'
        weight_decay, the amount of L2 regularization
    """
    #activations: 'relu', 'sigmoid', 'linear'
    def __init__(self, gamma = 0.001,  dimensions = [2,10,1], activations = ['relu','sigmoid'] ,weight_decay = 0):
        assert (len(dimensions)-1) == len(activations), "Number of dimensions and activation functions do not match"
        # number of layers of our MLP
        self.num_layers = len(dimensions)
        self.gamma = gamma
        self.weight_decay = weight_decay
        
        # initialize the weights
        self.weights = {}
        self.bias = {}
        # the first layer is the input data
        self.activations = {}
        self.activations_grad = {}
        
        for n in np.arange(self.num_layers - 1):
            # the weights are initialized acccording to a normal distribution and divided by the size of the layer they're on
            self.weights[n + 1] = np.random.randn(dimensions[n + 1],dimensions[n]) / np.sqrt(dimensions[n])
            # bias are all initialized to zero
            self.bias[n + 1] = np.zeros(dimensions[n + 1])
            
            if activations[n] == 'relu':
                self.activations[n+1] = self.relu
                self.activations_grad[n+1] = self.relu_gradient
            elif activations[n] == 'sigmoid':
                self.activations[n+1] = self.sigmoid
                self.activations_grad[n+1] = self.sigmoid_gradient
            else:
                self.activations[n+1] = lambda x : x
                self.activations_grad[n+1] = lambda x : 1
    
    
    #compute forward pass for an example
    def feed_forward(self, x):        
        # keep track of all z and a to compute gradient in the backpropagation
        z = {}
        # the first layer is the input data
        a = {1:x}
        # We compute z[n+1] = a[n] * w[n] + b[n]
        # and a[n+1] = f(z[n+1]) = f(a[n] * x[n] + b[n]) where * is the inner product
        for n in np.arange(1, self.num_layers):
            z[n + 1] = self.weights[n] @ a[n] + self.bias[n]
            a[n + 1] = self.activations[n](z[n + 1])
        y_pred = a[n+1]    
        return y_pred,a, z
    
    # return predictions for input examples
    def predict(self, X):
        preds = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_i_proba,_,_ = self.feed_forward(X[i].squeeze()) 
            preds[i] = (y_i_proba > 0.5)
        return preds
    
    # return estimated probabilities for input examples
    def predict_proba(self, X):
        preds = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_i_proba,_,_ = self.feed_forward(X[i].squeeze()) 
            preds[i] = y_i_proba
        return preds
    
    # compute gradients using backpropagation
    def back_propagate(self, y,y_pred, a, z):
        
        weights_gradient = {}
        bias_gradient = {}
        
        nabla = self.BCE_gradient(y,y_pred)
        
        for n in np.flip(np.arange(1, self.num_layers)):
            nabla = nabla * self.activations_grad[n](z[n+1])
            weights_gradient[n] = np.outer(nabla, a[n])
            bias_gradient[n] = nabla
            nabla = nabla @ self.weights[n]
        
        return weights_gradient, bias_gradient
    
    # performs a gradient descent step and apply L2 regularization
    def gradient_descent_step(self, weights_gradient, bias_gradient):
        for n in np.arange(1, self.num_layers):
            self.weights[n] = self.weights[n] - self.gamma * (weights_gradient[n] + self.weight_decay*self.weights[n])
            self.bias[n] = self.bias[n] - self.gamma * (bias_gradient[n] + self.weight_decay*self.bias[n])            
    
    def train(self, X, y, max_iter, batch_size = 1, decay = False, decay_rate = 3, decay_iteration = 0):
        """
        Main function of the MLP. It performs max_iter stochastic gradient descent steps with batch_size 1.
        At each iteration, it samples an example in X uniformly at random, computes the forward pass, and updates
        the weights using the gradients computed by the backward pass.
        
        input
            X, the samples used for training
            y, the corresponding labels
            max_iter, the number of gradient descent steps
            batch_size, number of samples on which to compute the gradient
            decay, whether to use learning rate decay
            decay_rate, the factor by which the learning rate is decays
            decay_iteration, every how many steps the learning rate is decayed
        output
            loss, the binary cross-entropy loss
        """
        for i in range(max_iter):
            if (decay):
                if ((i % decay_iteration == 0) and (i != 0)):
                    print("Iteration: {}".format(i))
                    print("Decay, lr : {}".format(self.gamma))
                    self.gamma = self.gamma/decay_rate
                    print("Decay, lr : {}".format(self.gamma))
                    print("")
            idxs = np.random.randint(0, X.shape[0],batch_size)
            X_batch = X[idxs].squeeze()
            y_batch = y[idxs]
            y_pred,a, z = self.feed_forward(X_batch)
            weights_gradient, bias_gradient = self.back_propagate(y_batch,y_pred,a, z)
            self.gradient_descent_step(weights_gradient, bias_gradient)
            if ((i % int(max_iter/5)) == 0):
                loss = self.BCE_loss(X,y)
                print("Iteration : {}, loss : {}".format(i,loss))
        loss = self.BCE_loss(X,y)
        print("Iteration : {}, loss : {}".format(i,loss))
        return loss
    
    def sigmoid_gradient(self,z):
        return sigmoid(z) * (1 - sigmoid(z))

    def relu(self,z):
        return np.where(z < 0, 0, z)

    def relu_gradient(self, z):
        return np.where(z < 0, 0, 1)

    def BCE_loss(self,X, y):
        # eps is added for numerical stability in the log
        loss = 0
        N = len(y)
        eps = 1e-7
        for i in range(N):
            y_pred,_,_ = self.feed_forward(X[i])
            loss_i = -(y[i]*np.log(y_pred+eps) + (1-y[i])*np.log(1-y_pred+eps))
            loss = loss + loss_i/N
        return loss

    def BCE_gradient(self,y,y_pred):
        # eps is added for numerical stability in the log
        eps = 1e-7
        return (-y/(y_pred+eps) + (1-y)/(1-y_pred+eps))
    
    def sigmoid(self,z):
        return 1.0/(1 + np.exp(-z))
    
   
