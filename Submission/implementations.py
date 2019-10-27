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

def preproc(X):
    col_means = np.nanmean(X, axis=0)
    col_means = np.nanmedian(X, axis=0)
    idxs = np.where(np.isnan(X))
    X[idxs] = np.take(col_means, idxs[1])

    #feature 1: correlations der_mass_MMC
    X_gt_mmc = np.array(X[:,0], copy=True)
    # X_0_cop = np.array(X[:,0], copy=True)
    X_gt_mmc[X_gt_mmc <= 140] = 140
    # X = np.column_stack((X, X_gt_mmc))
    X[:,0][X[:,0] > 140] = 140
    X = np.column_stack((X, X_gt_mmc))

    #feature 2: add momentums
    #tau momentum
    tau_px = X[:,13]*np.cos(X[:,15])
    tau_py = X[:,13]*np.sin(X[:,15])
    tau_pz = X[:,13]*np.sinh(X[:,14])
    tau_mod = X[:,13]*np.cosh(X[:,14])
    X = np.column_stack((X, tau_px,tau_py,tau_pz))
    #lep momentum
    lep_px = X[:,16]*np.cos(X[:,18])
    lep_py = X[:,16]*np.sin(X[:,18])
    lep_pz = X[:,16]*np.sinh(X[:,17])
    lep_mod = X[:,16]*np.cosh(X[:,17])
    X = np.column_stack((X, lep_px,lep_py,lep_pz))
    #leading jet momentum
    jet_px = X[:,23]*np.cos(X[:,25])
    jet_py = X[:,23]*np.sin(X[:,25])
    jet_pz = X[:,23]*np.sinh(X[:,24])
    jet_mod = X[:,23]*np.cosh(X[:,24])
    X = np.column_stack((X, jet_px,jet_py,jet_pz))
    #subleading jet momentum
    subjet_px = X[:,26]*np.cos(X[:,28])
    subjet_py = X[:,26]*np.sin(X[:,28])
    subjet_pz = X[:,26]*np.sinh(X[:,27])
    subjet_mod = X[:,26]*np.cosh(X[:,27])
    X = np.column_stack((X, subjet_px,subjet_py,subjet_pz))

    #feature 8: total invariant mass
    term_1 = np.sqrt(tau_px**2 + tau_py**2 + tau_pz**2) + np.sqrt(lep_px**2 + lep_py**2 + lep_pz**2) \
    + np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2) + np.sqrt(subjet_px**2 + subjet_py**2 + subjet_pz**2)
    term_2 = (tau_px + lep_px + jet_px + subjet_px)**2 + (tau_py + lep_py + jet_py + subjet_py)**2 \
            + (tau_pz + lep_pz + jet_pz + subjet_pz)**2
    inv_mass = np.sqrt(term_1**2 - term_2)

    #feature 9: log
    inv_log_cols = (0,1,2,3,4,5,7,8,9,10,12,13,16,19,21,23,26)
    X_inv_log_cols = np.log(1 / (1 + X[:, inv_log_cols]))
    X = np.hstack((X, X_inv_log_cols))
    # X_test_inv_log_cols = np.log(1 / (1 + X_test[:, inv_log_cols]))
    # X_test = np.hstack((X_test, X_test_inv_log_cols))


    #feature 4: categorical PRI_jet_num
    jet_num_0 = (X[:,22] == 0).astype(int)
    jet_num_1 = (X[:,22] == 1).astype(int)
    jet_num_2 = (X[:,22] == 2).astype(int)
    jet_num_3 = (X[:,22] == 3).astype(int)

    # #feature 5: pt ratios
    # #tau_lep_ratio = PRI_tau_pt/PRI_lep_pt
    tau_lep_ratio = X[:,13]/X[:,16]
    # #met_tot_ratio = PRI_met/PRI_met_sumet
    met_tot_ratio = X[:,19]/X[:,21]
    # X = np.column_stack((X, tau_lep_ratio,jets_ratio,met_tot_ratio))
    X = np.column_stack((X, tau_lep_ratio,met_tot_ratio))

    # #feature 6: jets_diff_angle
    jets_diff_angle = np.cos(X[:,24]-X[:,27])
    X = np.column_stack((X, jets_diff_angle))

    X = make_features(X)
    X = np.column_stack((X, jet_num_0, jet_num_1, jet_num_2, jet_num_3))
    return X

def make_features(X):
    # converting -999. to nan to use np.nanmean and np.nanstd
    X = np.where(X == -999., np.nan, X)
    # standardizing the data Xd = (X_d - E[X_d])/(std(X_d))
    X, means, stds = standardize(X)
    # since data is standirdized, the mean is more or less 0 for each feature so replacing by zero is reasonable and helps computations
    X = np.where(np.isnan(X), 0, X)
    # adding the 1 padding
    return np.column_stack((np.ones(X.shape[0]), X))

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

#feature 4: ratios
def make_feature_ratios(X):
    tau_lep_ratio = X[:,13]/X[:,16]
    met_tot_ratio = X[:,19]/X[:,21]
    X = np.column_stack((X, tau_lep_ratio,met_tot_ratio))
    return X

#feature 5: jets_diff_angle
def make_feature_diff_angles(X):
    jets_diff_angle = np.cos(X[:,24]-X[:,27])
    X = np.column_stack((X, jets_diff_angle))
    return X

class MLP:
    """
    Given activation functions and layer sizes, creates an instance of a Multi Layered Perceptrion (MLP), 
    using BCE as cost function and regularized stochastic gradient descent with batch size one.
    
    input
        gamma, the learning rate of the gradient 
        dimensions, the dimensions of the layers (Note that the input and the output shape must also be given)
        activations, the activation functions between the layers, Two possible: relu and sigmoid.
        weight_decay, the penalizing weight factor
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
    
    # returns a prediction
    def predict(self, X):
        preds = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_i_proba,_,_ = self.feed_forward(X[i].squeeze()) 
            preds[i] = (y_i_proba > 0.5)
        return preds
    
    def predict_proba(self, X):
        preds = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_i_proba,_,_ = self.feed_forward(X[i].squeeze()) 
            preds[i] = y_i_proba
        return preds
    
    # performs the backpropagation computed using the chain rule of derivation
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
    
    # performs a gradient descent step w_(t+1) = w_t - (grad_w_t +lambda) * gamma 
    def gradient_descent_step(self, weights_gradient, bias_gradient):
        for n in np.arange(1, self.num_layers):
            self.weights[n] = self.weights[n] - self.gamma * (weights_gradient[n] + self.weight_decay*self.weights[n])
            self.bias[n] = self.bias[n] - self.gamma * (bias_gradient[n] + self.weight_decay*self.bias[n])            
    
    def train(self, X, y, max_iter, batch_size = 1, decay = False, decay_rate = 3, decay_iteration = 0):
        """
        Main function of the MLP. It performs max_iter regularized stochastic gradient descent steps with batch_size 1.
        This means that at each iteration, it will randomly select a sample in X, compute its prediction (feedforward),
        then compute its gradient accross the whole network (backpropagation) and update all the weights of all layers.
        
        input
            X, the samples used for training
            y, the corresponding labels
            max_iter, the number of gradient descent steps
            batch_size, number of samples on which to compute the gradient
            decay, 
            decay_rate,
            decay_iteration,
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
    
   