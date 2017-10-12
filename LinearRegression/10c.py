# ## Ridge Regression with K-fold Cross validation

import numpy as np
import scipy.io as scio


data = scio.loadmat('HW1_Data/dataset2')

def transform(X, degree):
    Z = A = X
    for i in range(degree-1):
        Z = Z * A
        X = np.insert(X, [i+1], Z, axis=1)
    return np.insert(X, 0, 1, axis=1)


def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


def MSE(y, y_pred):
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)
    return np.mean((y - y_pred) ** 2)


def ridgeClosedForm(X, y, l):
    lI = l * np.identity(X.shape[1])
    return (X.T * X + lI).I * (X.T * y)


def ridgeGradientDescent(X, y, theta, alpha, minibatch_size, threshold, l):
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
                term = np.multiply(error, X_batch[:,j]) + np.multiply(l, theta)
                temp[0,j] = theta[0,j] - ((alpha / len(X_batch)) * np.sum(term))

            theta = temp
            
        cost.append(computeCost(X, y, theta))
        if cost[-2] - cost[-1] < threshold and cost[-2] - cost[-1] > 0:
            break
            
#         if i % 50 == 0:
#             print("Loss iter",i,": ",cost[-1])
#         i += 1
        
        
    return theta, cost


# In[25]:

def kFoldCV(X, y, fold):
    lam = np.arange(0.01, 0.1, 0.01)
    cost_lam = {}
    for l in lam:
        cost = 0
        for i in range(0,fold-1):
            data = np.concatenate((X, y), axis=1)
            data_split = np.split(data, fold)
            data1 = data_split[:i]
            data2 = data_split[i+1:]
            if not data1:
                data_train = np.concatenate((data2[:]), axis=0)
            else:
                data_train = np.concatenate((data1,data2), axis=0)
                data_train = np.concatenate((data_train[:]), axis=0)
            data_hold = data_split[i]
            cols = data_train.shape[1]
            X_train = np.matrix(data_train[:, 0:cols - 1])
            y_train = np.matrix(data_train[:, cols - 1:cols])
            theta = ridgeClosedForm(X_train, y_train, l)
            X_hold = np.matrix(data_hold[:, 0:cols - 1])
            y_hold = np.matrix(data_hold[:, cols - 1:cols])
            theta = theta.T
            cost = cost + computeCost(X_hold, y_hold, theta)
        avg_cost= cost/fold
        cost_lam[l] = avg_cost
        best_lam = min(cost_lam, key=lambda k: cost_lam[k])
    return best_lam


# In[30]:

def printer(X_train, y_train, X_test, y_test, fold, teta, size, threshold, degree, alpha):
    print("For K="+str(fold))
    lam = kFoldCV(X_train, y_train, fold)
    theta_2_closed_form = ridgeClosedForm(X_train, y_train, lam)
    theta_2_gd, cost_2_gd = ridgeGradientDescent(X_train, y_train, teta, alpha, size, threshold, lam)
    print("theta closed form", theta_2_closed_form.T)
    print("theta gradient descent", theta_2_gd)
    print("Lambda for degree k=2:", lam)
    print("-------TRAINING SET-------")
    print("cost closed form", computeCost(X_train, y_train, theta_2_closed_form.T))
    print("cost gradient descent", cost_2_gd[-1])
    print("MSE closed form", MSE(y_train, (X_train * theta_2_closed_form)))
    print("MSE train gradient descent", MSE(y_train, (X_train * theta_2_gd.T)))
    print("-------TEST SET-------")
    print("MSE closed form", MSE(y_test, (X_test * theta_2_closed_form)))
    print("MSE test gradient descent", MSE(y_test, (X_test * theta_2_gd.T)))
    print("-------------------------------------------------------------------")



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

    y_train = np.matrix(y_train)
    y_test = np.matrix(y_test)

    theta0_2 = np.matrix(np.zeros(3))
    theta0_3 = np.matrix(np.zeros(4))
    theta0_5 = np.matrix(np.zeros(6))

    batchSize = 15
    threshold = 0.001
    print("-------FOR DEGREE 2------")
    printer(X_train_2, y_train, X_test_2, y_test, 2, theta0_2, batchSize, threshold, 2, 0.0001)
    printer(X_train_2, y_train, X_test_2, y_test, 10, theta0_2, batchSize, threshold, 2, 0.0001)
    printer(X_train_2, y_train, X_test_2, y_test, X_train_2.shape[0], theta0_2, batchSize, threshold, 2, 0.0001)

    print("\n\n")
    print("-------FOR DEGREE 3------")
    printer(X_train_3, y_train, X_test_3, y_test, 2, theta0_3, batchSize, threshold, 3, 0.00001)
    printer(X_train_3, y_train, X_test_3, y_test, 10, theta0_3, batchSize, threshold, 3, 0.00001)
    printer(X_train_3, y_train, X_test_3, y_test, X_train_3.shape[0], theta0_3, batchSize, threshold, 3, 0.00001)

    print("\n\n")
    print("-------FOR DEGREE 5------")
    printer(X_train_5, y_train, X_test_5, y_test, 2, theta0_5, batchSize, 0.0008, 5, 0.000000001)
    printer(X_train_5, y_train, X_test_5, y_test, 10, theta0_5, batchSize, 0.0008, 5, 0.000000001)
    printer(X_train_5, y_train, X_test_5, y_test, X_train_5.shape[0], theta0_5, batchSize, 0.0008, 5, 0.000000001)


if __name__ == "__main__":
    main()


