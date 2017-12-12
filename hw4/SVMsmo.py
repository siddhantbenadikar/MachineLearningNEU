
# Support Vector Machine: SMO

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn.svm import LinearSVC
from skimage.transform import resize


# Calculates the dual of the Error
def dual(K, y, b, alpha, i):
    temp = np.multiply(alpha, y)
    return temp.T * K[:, i] + b


# Computes Eta value
def compute_eta(i, j, K):
    return 2 * K[i, j] - K[i, i] - K[j, j]


# Helper to calculate new alphaj value
def new_alphaj_value(alpha, y, Ei, Ej, eta, L, H):
    temp = alpha - (y * (Ei - Ej)) / eta
    if temp > H:
        return H
    elif temp < L:
        return L
    else:
        return temp


# Helper to compute b
def compute_b(b1, b2, alphai, alphaj, C):
    if 0 < alphai and alphai < C:
        return b1
    elif 0 < alphaj and alphaj < C:
        return b2
    else:
        return (b1 + b2) / 2


# Simplified smo algorithm
def smo(X, y, C, tolerance, max_passes, K):
    X = X[: , 1:]
    alpha = np.matrix(np.zeros(X.shape[0]))
    alpha = alpha.T
    b = 0
    passes = 0
    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(len(X)):
            Ei = dual(K, y, b, alpha, i) - y[i]
            if (y[i] * Ei < -tolerance and alpha[i] < C) or (y[i] * Ei > tolerance and alpha[i] > 0):
                j = np.random.choice(len(X))
                while j == i:
                    j = np.random.choice(len(X))

                Ej = dual(K, y, b, alpha, j) - y[j]
                old_alpha_i = alpha.item(i)
                old_alpha_j = alpha.item(j)

                if y[i] != y[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])

                if L == H:
                    continue

                eta = compute_eta(i, j, K)
                if eta >= 0:
                    continue

                alpha[j] = new_alphaj_value(alpha[j], y[j], Ei, Ej, eta, L, H)

                if abs(alpha[j] - old_alpha_j) < 0.00001:
                    continue

                alpha[i] = alpha[i] + (y[i] * y[j] * (old_alpha_j - alpha[j]))

#                 # Compute b1 and b2
#                 b_1 = b - Ei - y[i]*(alpha[i] - old_alpha_i)* K[i,i] - y[j] *(alpha[j] - old_alpha_j)*K[i,j]
#                 b_2 = b - Ej - y[i]*(alpha[i] - old_alpha_i)* K[i,j] - y[j] *(alpha[j] - old_alpha_j)*K[j,j]

#                 b = b_1 if (alpha[i]>0 and alpha[i]<C) else(b_2 if (alpha[j]>0 and alpha[j]<C) else (b_1+b_2)/2)

                b1 = b - Ei - y[i] * (alpha[i] - old_alpha_i) * K[i, i] - y[j] * (alpha[j] - old_alpha_j) * K[i, j]
                b2 = b - Ej - y[i] * (alpha[i] - old_alpha_i) * K[i, j] - y[j] * (alpha[j] - old_alpha_j) * K[j, j]
                b = compute_b(b1, b2, alpha[i], alpha[j], C)

                num_changed_alphas += 1
        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0
    return alpha, b


# Calculates theta
def calculate_theta_smo(X_train, y_train, alpha, b):
    temp1 = np.matrix(np.multiply(alpha,y_train))
    theta_smo = np.dot(temp1.T,X_train[:, 1:])
    theta_smo = np.insert(theta_smo, 0, b)
    return theta_smo


# plots the final output
def plotPrediction(X, y, theta, title = 'train set'):
    X = np.concatenate((X, y), axis=1)
    X = np.array(X)
    X1 = X[np.ix_(X[:, 3] == -1, (1,2))]
    X2 = X[np.ix_(X[:, 3] == 1, (1,2))]
    minimum = np.floor(X2.min())
    maximum = np.ceil(X2.max())
    plt.scatter(X1[:, 0], X1[:, 1], marker='+', color="blue", label="Class 0")
    plt.scatter(X2[:, 0], X2[:, 1], marker='o', color="green", label="Class 1")
    x = np.linspace(-3, 3, 10)
    theta=np.array(theta)
    slope = -theta[:,1]/theta[:,2]
    intercept = -theta[:, 0]/theta[:,2]
    y_pred = slope*x + intercept
    plt.ylim(minimum, maximum)
    plt.plot(x, y_pred, color='red', label="Boundary")
    plt.title(title)
    plt.legend()
    plt.show()


# predict new y values
def predict(theta, X):
    y_predict = np.sign((X[:,1:] * theta[:,1:].T) + theta[:,0])
    return y_predict


# calculate classification errors
def classificationError(y, y_pred):
    count = 0
    for i in range(len(y)):
        if y_pred[i] != y[i]:
            count += 1
    return count


# Prints classification error
def classification_error_print(X_train, y_train, theta):
    y_pred = predict(theta, X_train)
    classification_error = classificationError(y_train, y_pred)
    classification_error_percent =  classification_error / len(y_pred) * 100
    print("Miss-classified points:", classification_error)
    print("The classification error percent is: {:0.2f}%".format(classification_error_percent))
    print("The accuracy is: {:0.2f}%".format(100 - classification_error_percent))


def classification_error_print_sklearn(clf, X_train, y_train):
    y_pred = clf.predict(X_train)
    classification_error = classificationError(y_train, y_pred)
    classification_error_percent =  classification_error / len(y_pred) * 100
    print("Miss-classified points:", classification_error)
    print("The classification error percent is: {:0.2f}%".format(classification_error_percent))
    print("The accuracy is: {:0.2f}%".format(100 - classification_error_percent))


# def plot_c_vs_error():
#     train_err_list = []
#     test_err_list = []
#
#     for c in range(1,11):
#         # Calculate α,b for training data
#         alpha,b = smo(X_train, y_train, c, tolerance, max_passes)
#
#         # Calculate theta
#         temp = np.matrix(np.multiply(alpha,y_train))
#         theta_smo = np.dot(temp.T,X_train)
#         theta_smo[0,0] = b
#         theta_smo
#
#         # Mis-classification Error on Training
#         y_pred = predict(theta_smo, X_train)
#         misclassified = classificationError(y_train, y_pred)
#         err = misclassified / len(y_pred) * 100
#         train_err_list.append(err)
#
#         # Mis-classification Error on Test
#         y_pred = predict(theta_smo, X_test)
#         misclassified = classificationError(y_test, y_pred)
#         err = misclassified / len(y_pred) * 100
#         test_err_list.append(err)
#
#         fig, ax = plt.subplots(figsize=(12,8))
#
#         ax.plot(np.arange(1,11), train_err_list, '-', label='Training Error')
#         ax.plot(np.arange(1,11), test_err_list, '--', label='Test Error')
#
#         plt.legend()
#         fig.tight_layout()
#         plt.show()


def main():
    # Import data
    data = scio.loadmat('HW2_Data/data2')

    # Form testing and training data
    X_trn = np.insert(data['X_trn'], 0, 1, axis=1)
    y_trn = data['Y_trn']
    X_tst = np.insert(data['X_tst'], 0, 1, axis=1)
    y_tst = data['Y_tst']
    X_train = np.matrix(X_trn)
    y_train = np.matrix(y_trn)
    X_test = np.matrix(X_tst)
    y_test = np.matrix(y_tst)
    y_train = np.where(y_train == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)
    K = np.dot(X_train, X_train.T)

    # HYPER-PARAMETERS
    C = 10
    tolerance = 0.0001
    max_passes = 3
    alpha, b = smo(X_train, y_train, C, tolerance, max_passes, K)
    theta_smo = calculate_theta_smo(X_train, y_train, alpha, b)

    # Prints classification error
    print()
    print("<---------------CLASSIFICATION USING SIMPLIFIED SMO SVM--------------->")
    print("FOR TRAINING SET")
    classification_error_print(X_train, y_train, theta_smo)

    print()
    print("FOR TRAINING SET")
    classification_error_print(X_test, y_test, theta_smo)

    # Plots prediction
    plotPrediction(X_train, y_train, theta_smo, "training set")
    plotPrediction(X_test, y_test, theta_smo, "test set")

    #SKLEARN
    clf = LinearSVC(random_state=0)
    clf.fit(X_train, y_train)
    clf.score(X_train, y_train)
    print()
    print("<---------------CLASSIFICATION USING SKLEARN SVM--------------->")
    print("FOR TRAINING SET")
    classification_error_print_sklearn(clf, X_train, y_train)
    print()
    print("FOR TRAINING SET")
    classification_error_print_sklearn(clf, X_test, y_test)
    theta_smo_sklearn = clf.coef_

    plotPrediction(X_train, y_train, theta_smo_sklearn, "train set")
    plotPrediction(X_test, y_test, theta_smo_sklearn, "test set")


def load_data(path, col_name):
    resize_width = 17
    resize_height = 20

    ip = scio.loadmat(path)

    N = ip[col_name].shape[1] * ip[col_name][:, 0][0].shape[2]
    num_labels = ip[col_name].shape[1]

    size = (resize_height, resize_width)
    X = np.zeros((N, resize_height * resize_width))
    Y = np.full((N, num_labels), -1)

    img_index = 0

    for i in range(num_labels):
        curr_class_data = ip[col_name][:, i][0]
        for j in range(curr_class_data.shape[2]):
            img_resized = resize(curr_class_data[:, :, j], size, mode='constant')
            X[img_index, :] = img_resized.flatten()
            Y[img_index, i] = 1
            img_index += 1

    return X, Y


def execute3b():
    # ----------------------- Train the 10 classifiers -------------------------

    C = 10
    tolerance = 0.0001
    max_passes = 3

    path = "ExtYaleB10.mat"
    col_name = 'train'

    X_trn, Y_trn = load_data(path, col_name)
    X_trn = np.matrix(X_trn)
    K = np.dot(X_trn, X_trn.T)

    classifiers = []
    for i in range(10):
        print("Training classifier #", i)
        curr_Y_trn = np.matrix(np.reshape(Y_trn[:, i], (Y_trn.shape[0], 1)))
        alpha, b = smo(X_trn, curr_Y_trn, C, tolerance, max_passes, K)
        theta_smo = calculate_theta_smo(X_trn, curr_Y_trn, alpha, b)
        classifiers.append(theta_smo)

    # ---------------------- Run the model on test data ------------------------

    col_name = "test"

    X_tst, Y_tst = load_data(path, col_name)
    X_tst = np.matrix(X_tst)

    for i in range(10):
        curr_Y_tst = np.matrix(np.reshape(Y_tst[:, i], (Y_tst.shape[0], 1)))

        theta_smo = classifiers[i]
        classification_error_print(X_tst, curr_Y_tst, theta_smo)

def execute3d(X_trn):
    # ----------------------- Train the 10 classifiers -------------------------

    C = 10
    tolerance = 0.0001
    max_passes = 3

    path = "ExtYaleB10.mat"
    col_name = 'train'

    X_train, Y_trn = load_data(path, col_name)
    K = np.dot(X_trn, X_trn.T)

    classifiers = []
    for i in range(10):
        print("Training classifier #", i)
        curr_Y_trn = np.matrix(np.reshape(Y_trn[:, i], (Y_trn.shape[0], 1)))
        alpha, b = smo(X_trn, curr_Y_trn, C, tolerance, max_passes, K)
        theta_smo = calculate_theta_smo(X_trn, curr_Y_trn, alpha, b)
        classifiers.append(theta_smo)

    # ---------------------- Run the model on test data ------------------------

    col_name = "test"

    X_tst, Y_tst = load_data(path, col_name)
    X_tst = np.matrix(X_tst)

    for i in range(10):
        curr_Y_tst = np.matrix(np.reshape(Y_tst[:, i], (Y_tst.shape[0], 1)))

        theta_smo = classifiers[i]
        classification_error_print(X_tst, curr_Y_tst, theta_smo)

if __name__ == "__main__":
    execute3b()