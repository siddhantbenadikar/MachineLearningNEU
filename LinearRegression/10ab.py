# ## LINEAR REGRESSION

import numpy as np
import scipy.io as scio

data = scio.loadmat('HW1_Data/dataset1')


def transform(X, degree):
    Z = A = X
    for i in range(degree-1):
        Z = Z * A
        X = np.insert(X, [i+1], Z, axis=1)
    return np.insert(X, 0, 1, axis=1)


def MSE(y, y_pred):
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)
    return np.mean((y - y_pred) ** 2)


def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


def closedFormEquation(X, y):
    return (X.T * X).I * (X.T * y) 


def stochasticGradientDescent(X, y, theta, alpha, minibatch_size, threshold):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = theta.ravel().shape[1]
    cost = [np.inf]
    i = 1

    while True:
        for k in range(len(X) // minibatch_size):
            index_list = np.random.choice(len(X), size=minibatch_size, replace=False)
            X_batch = X[index_list]
            y_batch = y[index_list]
            error = (X_batch * theta.T) - y_batch
            
            for j in range(parameters):
                term = np.multiply(error, X_batch[:,j])
                temp[0,j] = theta[0,j] - ((alpha / len(X_batch)) * np.sum(term))

            theta = temp
            
        cost.append(computeCost(X, y, theta))
        if cost[-2] - cost[-1] < threshold and cost[-2] - cost[-1] > 0:
            break
            
#         if i % 50 == 0:
#             print("Loss iter",i,": ",cost[-1])
#         i += 1
        
        
    return theta, cost



def main():
    X_train = data['X_trn']
    y_train = data['Y_trn']
    X_test = data['X_tst']
    y_test = data['Y_tst']

    X_train_2 = np.matrix(transform(X_train, 2))
    X_train_3 = np.matrix(transform(X_train, 3))
    X_train_5 = np.matrix(transform(X_train, 5))

    X_test_2 = np.matrix(transform(X_test, 2))
    X_test_3 = np.matrix(transform(X_test, 3))
    X_test_5 = np.matrix(transform(X_test, 5))

    # In[163]:

    y_train = np.matrix(y_train)
    y_test = np.matrix(y_test)

    theta0_2 = np.matrix(np.zeros(3))
    theta0_3 = np.matrix(np.zeros(4))
    theta0_5 = np.matrix(np.zeros(6))

    batchSize = 10
    threshold = 0.0001

    # ## For degree 2
    print("FOR DEGREE 2")
    theta_2_closed_form = closedFormEquation(X_train_2, y_train)
    theta_2_gd, cost_2_gd = stochasticGradientDescent(X_train_2, y_train, theta0_2, 0.01, batchSize, threshold)
    print("theta closed form", theta_2_closed_form.T)
    print("theta gradient descent", theta_2_gd)
    print("-------TRAINING SET-------")
    print("cost closed form", computeCost(X_train_2, y_train, theta_2_closed_form.T))
    print("cost gradient descent", cost_2_gd[-1])
    print("MSE closed form", MSE(y_train, (X_train_2 * theta_2_closed_form)))
    print("MSE train gradient descent", MSE(y_train, (X_train_2 * theta_2_gd.T)))
    print("-------TEST SET-------")
    print("MSE closed form", MSE(y_test, (X_test_2 * theta_2_closed_form)))
    print("MSE test gradient descent", MSE(y_test, (X_test_2 * theta_2_gd.T)))
    print()
    print()

    #  For degree 3
    print("FOR DEGREE 3")
    theta_3_closed_form = closedFormEquation(X_train_3, y_train)
    theta_3_gd, cost_3_gd = stochasticGradientDescent(X_train_3, y_train, theta0_3, 0.0001, batchSize, threshold)
    print("theta closed form", theta_3_closed_form.T)
    print("theta gradient descent", theta_3_gd)
    print("-------TRAINING SET-------")
    print("cost closed form", computeCost(X_train_3, y_train, theta_3_closed_form.T))
    print("cost gradient descent", cost_3_gd[-1])
    print("MSE closed form", MSE(y_train, (X_train_3 * theta_3_closed_form)))
    print("MSE train gradient descent", MSE(y_train, (X_train_3 * theta_3_gd.T)))
    print("-------TEST SET-------")
    print("MSE closed form", MSE(y_test, (X_test_3 * theta_3_closed_form)))
    print("MSE test gradient descent", MSE(y_test, (X_test_3 * theta_3_gd.T)))
    print()
    print()

    # For degree 5

    print("FOR DEGREE 5")
    theta_5_closed_form = closedFormEquation(X_train_5, y_train)
    theta_5_gd, cost_5_gd = stochasticGradientDescent(X_train_5, y_train, theta0_5, 0.00001, batchSize, threshold)
    print("theta closed form", theta_5_closed_form.T)
    print("theta gradient descent", theta_5_gd)
    print("-------TRAINING SET-------")
    print("cost closed form", computeCost(X_train_5, y_train, theta_5_closed_form.T))
    print("cost gradient descent", cost_5_gd[-1])
    print("MSE closed form", MSE(y_train, (X_train_5 * theta_5_closed_form)))
    print("MSE train gradient descent", MSE(y_train, (X_train_5 * theta_5_gd.T)))
    print("-------TEST SET-------")
    print("MSE closed form", MSE(y_test, (X_test_5 * theta_5_closed_form)))
    print("MSE test gradient descent", MSE(y_test, (X_test_5 * theta_5_gd.T)))


if __name__ == "__main__":
    main()