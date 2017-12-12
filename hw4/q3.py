from hw4 import FF_NeuralNet as NN
from hw4 import SVMsmo as svm
from hw4 import pca as pca
from hw4 import Clustering as cls
from hw4 import auto_encoder as enc
from hw4 import logstic as log
import matplotlib.pyplot as plt
import numpy as np


# Run to get output for ans 3a, you can change activation fucnti as
# 'tanh', 'sigmoid', 'relu'. You can set print loss to True to view
# loss per iteration
def a():
    set_model = NN.FNN(activation='sigmoid')
    X_test, y_test = set_model.load_data('test')
    weights, loss, activations = set_model.fit(5, print_loss=False)
    acc = set_model.accuracy(weights, X_test, y_test)
    print('Test accuracy: {:0.2f}%'.format(acc))
    plt.plot(range(20000), loss)
    plt.show()

    ## NOTE: run the main() in FF_NeuralNet.py to get values for different number of
    # hidden nodes


# Warning: This take alot of time. At least on my machine
def b():
    svm.execute3b()


def c():
    log.execute_3c(alpha=0.01, folds=10)

def d():
    # Perform PCA
    X_train, y_train = svm.load_data('ExtYaleB10.mat', 'train')
    X_train = X_train.T
    U, mean, X_pca = pca.pca(X_train, 100)

    #NN
    set_model = NN.FNN(X=X_pca.T, activation='relu')
    X_test, y_test = set_model.load_data('test')
    X_test = X_test.T
    U, mean, X_pca_test = pca.pca(X_test, 100)

    i = 3
    ac = []
    while i < 34:
        weights, loss, scores = set_model.fit(i, print_loss=False)
        acc = set_model.accuracy(weights, X_pca_test.T, y_test)
        ac.append(acc)
        print('FOR I=', i, 'Test accuracy: {:0.2f}%'.format(acc))
        i += 3
    plt.plot([x * 3 for x in range(len(ac))], ac)
    plt.show()


# auto encoder to reduce d = 100
def e():
    set_model = enc.AutoEncoder(activation='relu')
    weights, errors, encoded_layer = set_model.fit(100, print_mse=False)
    set_model_FNN = NN.FNN(X=encoded_layer, activation='relu')
    X_train, y_train = set_model_FNN.load_data('train')
    i = 3
    ac = []
    while i < 34:
        weights, loss, scores = set_model_FNN.fit(i, print_loss=False)
        acc = set_model.accuracy(weights, encoded_layer, y_train)
        ac.append(acc)
        print('FOR I=', i, 'Test accuracy: {:0.2f}%'.format(acc))
        i += 3
    plt.plot([x * 3 for x in range(len(ac))], ac)
    plt.show()


def f():
    X_train, y_train = svm.load_data('ExtYaleB10.mat', 'train')
    X = X_train.T
    U, mean, X_pca = pca.pca(X, 2)

    pca.plotCluster(X_pca, 0, 1)

def g():
    X_train, y_train = svm.load_data('ExtYaleB10.mat', 'train')
    X = X_train.T
    U, mean, X_pca = pca.pca(X, 100)
    Z = cls.k_means(X_pca, 10, 15)
    output = []
    # Get the output values from Z
    for key in Z:
        output.append(Z[key])
    U, mean, X_pca = pca.pca(X, 2)
    cls.plot_data(X_pca.T, output)

def h():
    X_train, y_train = svm.load_data('ExtYaleB10.mat', 'train')
    U, mean, X_pca = pca.pca(X_train, 2)
    # Spectral Clustering
    W = cls.generate_W(X_train, 10, 25)
    Z_s = cls.spectral(W, 10)
    output = []
    for key in Z_s:
        output.append(Z_s[key])
    count = 0
    for j in range(10):
        curr_Y_tst = np.matrix(np.reshape(y_train[:, j], (y_train.shape[0], 1)))
        for i in range(len(output)):
            if output[i] != curr_Y_tst[i]:
                count += 1
    print(count)
    cls.plot_data(X_pca.T, output)


if __name__ == '__main__':
    a()
