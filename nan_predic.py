import numpy as np
from implementations import *
from proj1_helpers import *

def nan_predic(X_):
    X_ = np.where(X_ == -999., np.nan, X_)
    #mass prediction

    mass_nan_index = [0]
    jet_nan_indexes = [4,5,6,12,26,27,28]
    jet_sub_nan_indexes = [23,24,25]

    X_noNans = np.delete(X_, np.concatenate((jet_nan_indexes, jet_sub_nan_indexes)), 1)

    defined_mass_idx = np.where(np.isfinite(X_noNans[:,0])) #indexes of samples with a mass
    undefined_mass_idx = np.where(np.isnan(X_noNans[:,0]))  #indexes of samples with a missing mass

    X_mass = X_noNans[defined_mass_idx][:,1:] # we select the X of training set
    y_mass = X_noNans[defined_mass_idx][:,:1] # we select the y of the taining set

    X_mass,_,_ = standardize(X_mass) # we standardize both 
    y_mass,_,_ = standardize(y_mass)
    y_mass = y_mass.squeeze()

    initial_w = np.ones(X_mass.shape[1]) / X_mass.shape[1]
    lambda_ = 0.0001
    max_iter = 1000
    gamma = 0.005
    w, loss = least_squares_GD(y_mass, X_mass, initial_w, max_iter, gamma) # since it isn't classification, we use least_squares

    to_predict_mass,_,_ = standardize(X_noNans[undefined_mass_idx][:,1:]) # we select the set to be predicted
    predicted_masses = to_predict_mass @ w
    
    # we crop because we don't want to predict outside of the values
    predicted_masses = np.where(predicted_masses < np.min(y_mass), np.min(y_mass), predicted_masses)
    predicted_masses = np.where(predicted_masses > np.max(y_mass), np.max(y_mass), predicted_masses)

    masses = np.zeros(X_.shape[0])
    masses[defined_mass_idx] = y_mass
    masses[undefined_mass_idx] = predicted_masses

    X_[:,mass_nan_index[0]] = masses

    X_noNans = np.delete(X_, np.concatenate((mass_nan_index, jet_sub_nan_indexes)), 1)


    # jet predictions

    defined_jet_idx = np.where(np.isfinite(X_[:,jet_nan_indexes[0]]))[0] #indexes of samples with jets
    undefined_jet_idx = np.where(np.isnan(X_[:,jet_nan_indexes[0]]))[0]  #indexes of samples with missing jets

    X_jet = np.delete(X_, np.concatenate((mass_nan_index, jet_nan_indexes, jet_sub_nan_indexes)), 1)[defined_jet_idx] # we select the X of training set
    to_predict_jets = np.delete(X_, np.concatenate((mass_nan_index, jet_nan_indexes, jet_sub_nan_indexes)), 1)[undefined_jet_idx]
    y_jet_1 = X_[defined_jet_idx][:,jet_nan_indexes[0]] # we select the y of the taining set [4]
    y_jet_2 = X_[defined_jet_idx][:,jet_nan_indexes[1]] # we select the y of the taining set [5]
    y_jet_3 = X_[defined_jet_idx][:,jet_nan_indexes[2]] # we select the y of the taining set [6]
    y_jet_4 = X_[defined_jet_idx][:,jet_nan_indexes[3]] # we select the y of the taining set [12]
    y_jet_5 = X_[defined_jet_idx][:,jet_nan_indexes[4]] # we select the y of the taining set [26]
    y_jet_6 = X_[defined_jet_idx][:,jet_nan_indexes[5]] # we select the y of the taining set [27]
    y_jet_7 = X_[defined_jet_idx][:,jet_nan_indexes[6]] # we select the y of the taining set [28]

    X_jet,_,_ = standardize(X_jet) # we standardize
    to_predict_jets,_,_ = standardize(to_predict_jets)
    y_jet_1,_,_ = standardize(y_jet_1)
    y_jet_2,_,_ = standardize(y_jet_2)
    y_jet_3,_,_ = standardize(y_jet_3)
    y_jet_4,_,_ = standardize(y_jet_4)
    y_jet_5,_,_ = standardize(y_jet_5)
    y_jet_6,_,_ = standardize(y_jet_6)
    y_jet_7,_,_ = standardize(y_jet_7)

    y_jet_1 = y_jet_1.squeeze()
    y_jet_2 = y_jet_2.squeeze()
    y_jet_3 = y_jet_3.squeeze()
    y_jet_4 = y_jet_4.squeeze()
    y_jet_5 = y_jet_5.squeeze()
    y_jet_6 = y_jet_6.squeeze()
    y_jet_7 = y_jet_7.squeeze()

    y_jets = [y_jet_1, y_jet_2, y_jet_3, y_jet_4, y_jet_4, y_jet_5, y_jet_6, y_jet_7]

    lambda_ = 0.0001
    max_iter = 1000
    gamma = 0.005
    initial_w = np.ones(X_jet.shape[1]) / X_jet.shape[1]

    ws = []
    losses = []

    for y_jet in y_jets:
        w, loss = least_squares_GD(y_jet, X_jet, initial_w, max_iter, gamma) # since it isn't classification, we use least_squares
        ws.append(w)
        losses.append(loss)

    jet_predictions = []
   
    i = 0
    for w in ws:
        jet_prediction = to_predict_jets @ w
        jet_prediction = np.where(jet_prediction < np.min(y_jets[i]), np.min(y_jets[i]), jet_prediction)
        jet_prediction = np.where(jet_prediction > np.max(y_jets[i]), np.max(y_jets[i]), jet_prediction)
        i = i + 1
        jet_predictions.append(jet_prediction)

    for i in range(len(jet_nan_indexes)):
        jet_col = np.zeros(X_.shape[0])
        jet_col[defined_jet_idx] = y_jets[i]
        jet_col[undefined_jet_idx] = jet_predictions[i]
        X_[:, jet_nan_indexes[i]] = jet_col
        
    return X_