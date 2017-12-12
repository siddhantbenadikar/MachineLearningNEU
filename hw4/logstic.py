
# coding: utf-8

# Import necessary libraries

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as r
import scipy.io as sio
import math
from skimage.transform import resize

# helper functions
def load_data(path, col_name):

    resize_width = 17
    resize_height = 20
    
    ip = sio.loadmat(path)

    N = ip[col_name].shape[1] * ip[col_name][:, 0][0].shape[2]
    num_labels = ip[col_name].shape[1]

    size = (resize_height, resize_width)
    X = np.zeros((N, resize_height * resize_width))
    Y = np.zeros((N, num_labels))

    img_index = 0

    for i in range(num_labels):
        curr_class_data = ip[col_name][:,i][0]
        for j in range(curr_class_data.shape[2]):
            img_resized = resize(curr_class_data[:,:,j], size, mode='constant')
            X[img_index, :] = img_resized.flatten()
            Y[img_index, i] = 1
            img_index += 1
    
    # Add bias term to X
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))
    
    return X, Y

def plot_cv_error(l2_penality_values, l2_penality_mse):
    plt.plot(l2_penality_values, l2_penality_mse, 'k-')
    plt.xlabel('$\ell_2$ penalty')
    plt.ylabel('K-fold cross validation error')
    plt.xscale('log')
    plt.yscale('log')

def compute_prediction(W, H):

    Z = np.dot(H, W)  # Results in a N x 1 matrix    
    return 1 / (1 + (np.exp(-Z)))  # Predict

def compute_coefficients(W, H, Y, alpha, l2_penality):
    P = compute_prediction(W, H)  # Predict
    error = P - Y
    term = H.T * error
    return W - (alpha / H.shape[0]) * (term + l2_penality * W)  # Calculate coefficients

def compute_cost(H, Y, W, l2_penality):

    # Calcualte -(h(x) * w)
    prod = H * W
    prod = -(prod)
    
    # Calculate regularization term
    reg = np.sum(W * W.T)
    reg = (l2_penality / 2) * pow(reg, 0.5)
    
    # Calculate (1 - 1(y = +1)) * -(h(x) * w)
    first_term = np.multiply((1 - Y), prod)
    
    # Calculate ln(1 + e ^ -(h(x) * w))
    second_term = np.log(1 + np.exp(prod))
    
    # Calculate log likelihood
    cost = first_term - second_term
    
    # Calculate the overall cost
    return np.sum(cost, axis=0) + reg

def gradient_descent(H, Y, W, alpha, l2_penality):
    temp = np.matrix(np.zeros(W.shape))
    error = []
    cost = []
    i = 0
    
    while True:
        # Predict using W from previous iteration
        P = compute_prediction(W, H)
        P = vector_indicator(P)
        
        # Calculate the number of miscalculations and
        # the cost incurred
        error.append(np.sum(np.absolute(Y - P), axis=0).getA()[0][0])
        cost.append(compute_cost(H, Y, W, l2_penality).getA()[0][0])
        
        W = compute_coefficients(W, H, Y, alpha, l2_penality)        
        
        # Break out of the loop only when the cost
        # incurred is less than the threshold
        
        if ((i > 0) and 
            (cost[i] - cost[i - 1] < 0.005)):
            print("Iterations: ", i)            
            print("Miscalculations: ", error[i])
            print("Cost: ", cost[i])
            break
        
        i += 1
    
    return W, error, cost

def plot_data(data, W):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    x = np.linspace(min_vals[0], max_vals[0], 100)

    f = 0
    for i in range(len(W)):
        f += x * W.getA()[i][0]
    
    fig, ax = plt.subplots(figsize=(12,8))
    
    # Draw the plots
    ax.plot(x, f, 'r', label='Classifier')
    ax.scatter(data.x1, data.x2, label='Data', c=data[[y_col]])
    
    # Set extra properties for readability
    ax.legend(loc=2)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('x1 vs. x2')
    
    # Set the x and y axis limits for the plot
    x1_min = data.x1.min()
    x1_max = data.x1.max()
    ax.set_xlim(x1_min + 0.2 * x1_min, x1_max + 0.2 * x1_max)
    
    x2_min = data.x2.min()
    x2_max = data.x2.max()
    ax.set_ylim(x2_min + 0.2 * x2_min, x2_max + 0.2 * x2_max)

def indicator(score):
    return 1 if (score >= 0.5) else 0

vector_indicator = np.vectorize(indicator)


# ## K-Fold cross validation - Helper methods

# In[102]:

def generate_folds(X, k, randomize = False):
    # Shuffle the data if necessary    
    if randomize:
        X = list(X)
        r.shuffle(X)
    
    for i in range(k):
        training = [x for j, x in enumerate(X, start = 1) if j % k != i]
        validation = [x for j, x in enumerate(X, start = 1) if j % k == i]
        yield training, validation

def do_k_fold_CV(data, theta, alpha, l2_penality_values, K):
    # For each value of L2 penality, compute
    # the average MSE after fitting a model
    # for each fold
    l2_penality_mse = np.zeros(len(l2_penality_values))
    
    # Define variables to track the min MSE
    # and best L2 penality
    min_mse = None
    best_l2_penality = None
    
    # Create as many folds as needed for cross
    # validation and run logistic regression on
    # each training-validation pair
    fold_index = 0
    for training, validation in generate_folds(data, K, True):
        print("Optimizing with fold #", fold_index)
        
        # Convert into matrices
        training = np.matrix(training)
        validation = np.matrix(validation)
        
        # Split the datasets into X and Y
        t_X = training[:, :-1]
        t_Y = training[:, -1:]
        v_X = validation[:, :-1]
        v_Y = validation[:, -1:]
        
        for i, l2_penality in enumerate(l2_penality_values):
            print("Working with l2_penality: ", l2_penality)
            # Initialize theta
            t_theta, error, cost = gradient_descent(t_X, t_Y, theta, alpha, l2_penality)
            # Predict validation set outputs
            v_Y_prediction = compute_prediction(t_theta, v_X)
            #print("v_Y_prediction: ", v_Y_prediction.shape)
            v_l1_error = v_Y_prediction - v_Y
            #print("v_l1_error: ", v_l1_error.shape)
            v_l2_error = v_l1_error.T * v_l1_error
            #print("v_l2_error: ", v_l2_error.shape)
            l2_penality_mse[i] += v_l2_error.sum()
        
        fold_index += 1
    
    l2_penality_mse = l2_penality_mse / K
    
    # Find the min mse and corresponding l2 penality
    min_mse = None
    best_l2_penality = None
    
    for i, val in enumerate(l2_penality_mse):
        if min_mse is None or val < min_mse:
            min_mse = val
            best_l2_penality = l2_penality_values[i]            
    
    return l2_penality_mse, best_l2_penality


# ## PCA

# In[105]:

def pca(X, d):
    # Do mean normalization
    M_X = np.sum(X, axis = 0)
    M_X = M_X / X.shape[0]
    X = X - M_X

    # Find the correlation matrix
    C = np.dot(X.T, X) / X.shape[0]

    # Do eigenvalue decomposition and get hold of 
    # the eigenvalues (D) and eigenvectors (V) of 
    # covariance matrix
    D, V = np.linalg.eig(C)

    # Extract the top-d eigenvectors
    V = V[:, 0:d]
    
    # Represent data in this basis
    Y = np.dot(X, V)
    
    # Calculate the mean of low-dimensional space
    M_Y = np.sum(Y, axis=0) / Y.shape[0]
    
    return V, M_Y, Y


# In[108]:

def execute_3c(alpha, folds = None):
    path = "ExtYaleB10.mat"
    col_name = 'train'    

    X_trn, Y_trn = load_data(path, col_name)

    if folds is None:
        folds = X_trn.shape[0]

    # ----------------------- Train the 10 classifiers -------------------------
    classifiers = []
    for i in range(10):
        print("---------------- Training classifier #", i, " ---------------------")
        curr_Y = np.reshape(Y_trn[:, i], (Y_trn.shape[0], 1))
        train_data = np.hstack((X_trn, curr_Y))
        theta = np.matrix(np.zeros(X_trn.shape[1])).T

        # Do k-fold cross validation
        l2_penality_values = np.logspace(-2, 1, num=5)
        l2_penality_mse, best_l2_penality = do_k_fold_CV(train_data, theta, alpha, l2_penality_values, folds)    

        print("best_l2_penality: ", best_l2_penality)

        # Run gradient descent
        theta, error, cost = gradient_descent(X_trn, curr_Y, theta, alpha, best_l2_penality)
        classifiers.append(theta)

    # ---------------------- Run all models on test data -----------------------

    col_name = "test"

    X_tst, Y_tst = load_data(path, col_name)

    # Combine all columns of Y_tst into a single one
    u, v = Y_tst.nonzero()
    for i in range(len(v)):
        Y_tst[u, v] = v
    Y_tst = np.matrix(np.reshape(np.sum(Y_tst, axis=1), (Y_tst.shape[0], 1)))
    print(Y_tst.getA1())

    probs = []

    for i in range(10):
        print("Running classifier #", i, " on test data...")
        probs.append(compute_prediction(classifiers[i], X_tst))

    probs = np.hstack(probs)
    predicted_classes = np.matrix(probs.argmax(axis=1))

    compared = np.abs(Y_tst - predicted_classes)
    error = np.sum(np.where(compared != 0, 1, 0), axis=0)
    print("Number of miscalculations: ", error)

def execute_3d(alpha, folds = None):
    path = "ExtYaleB10.mat"
    col_name = 'train'    

    X_trn, Y_trn = load_data(path, col_name)
    
    print("Reducing the dimension of training data to 100 using PCA...")
    V_trn, M_trn, X_trn = pca(X_trn, 100)

    if folds is None:
        folds = X_trn.shape[0]

    # ----------------------- Train the 10 classifiers -------------------------
    classifiers = []
    for i in range(10):
        print("---------------- Training classifier #", i, " ---------------------")
        curr_Y = np.reshape(Y_trn[:, i], (Y_trn.shape[0], 1))
        train_data = np.hstack((X_trn, curr_Y))
        theta = np.matrix(np.zeros(X_trn.shape[1])).T

        # Do k-fold cross validation
        l2_penality_values = np.logspace(-2, 1, num=5)
        l2_penality_mse, best_l2_penality = do_k_fold_CV(train_data, theta, alpha, l2_penality_values, folds)    

        print("best_l2_penality: ", best_l2_penality)

        # Run gradient descent
        theta, error, cost = gradient_descent(X_trn, curr_Y, theta, alpha, best_l2_penality)
        classifiers.append(theta)

    # ---------------------- Run all models on test data -----------------------

    col_name = "test"

    X_tst, Y_tst = load_data(path, col_name)
    
    print("Reducing the dimension of test data to 100 using PCA...")
    V_tst, M_tst, X_tst = pca(X_tst, 100)

    # Combine all columns of Y_tst into a single one
    u, v = Y_tst.nonzero()
    for i in range(len(v)):
        Y_tst[u, v] = v
    Y_tst = np.matrix(np.reshape(np.sum(Y_tst, axis=1), (Y_tst.shape[0], 1)))
    
    probs = []

    for i in range(10):
        print("Running classifier #", i, " on test data...")
        probs.append(compute_prediction(classifiers[i], X_tst))

    probs = np.hstack(probs)
    predicted_classes = np.matrix(probs.argmax(axis=1))

    compared = np.abs(Y_tst - predicted_classes)
    error = np.sum(np.where(compared != 0, 1, 0), axis=0)
    print("Number of miscalculations: ", error)


