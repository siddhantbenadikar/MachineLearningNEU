import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from skimage.transform import resize

np.random.seed(0)


class FNN(object):
    def __init__(self, X=None, activation=None, output_dim=10, learning_rate=0.0001, lamda=0.01):
        if X is None:
            self.X, self.y = self.load_data('train')
        else:
            self.X = X
            temp, self.y = self.load_data('train')
        self.num_samples = len(self.X)
        self.input_dim = len(self.X[0])
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.lamda = lamda
        if activation == 'tanh':
            self.activation_derivative = self.tanh_derivative
            self.activation = self.tanh
        if activation == 'sigmoid':
            self.activation_derivative = self.sigmoid_derivative
            self.activation = self.sigmoid
        if activation == 'relu':
            self.activation_derivative= self.relu_derivative
            self.activation = self.relu

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1.0 - x ** 2

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def softmax(self, output_array):
        logits_exp = np.exp(output_array)
        return logits_exp / np.sum(logits_exp, axis=1, keepdims=True)

    # # Generating D x N matrix from the given data
    # Downsample it to 20 x 17 to reduce computation cost
    def load_data(self, data_type, height=20, width=17):

        new_height = height
        new_width = width

        mat = sio.loadmat('ExtYaleB10.mat')

        N = mat[data_type].shape[1] * mat[data_type][0][0].shape[2]
        num_labels = mat[data_type].shape[1]

        size = (new_height, new_width)
        X = np.zeros((N, new_height * new_width))
        Y = []

        img_index = 0
        flag = 0

        for i in range(num_labels):
            curr_class_data = mat[data_type][:, i][0]
            for j in range(curr_class_data.shape[2]):
                img_resized = resize(curr_class_data[:, :, j], size, mode='constant')
                X[img_index, :] = img_resized.flatten()
                Y.append(flag)
                img_index += 1
            flag += 1
        return X, Y

    # Calculates loss cross entropy
    def calculate_loss(self, weights):
        W1, b1, W2, b2 = weights['W1'], weights['b1'], weights['W2'], weights['b2']

        z1 = self.X.dot(W1) + b1
        a1 = self.activation(z1)
        z2 = a1.dot(W2) + b2
        scores = self.softmax(z2)

        log_probability = -np.log(scores[range(self.num_samples), self.y])
        data_loss = np.sum(log_probability)

        data_loss += self.lamda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

        return 1. / self.num_samples * data_loss

    # Predicts final probability distribution of belonging to that class
    def predict(self, weights, x):
        W1, b1, W2, b2 = weights['W1'], weights['b1'], weights['W2'], weights['b2']

        z1 = x.dot(W1) + b1
        a1 = self.activation(z1)
        z2 = a1.dot(W2) + b2
        scores = self.softmax(z2)

        probability = scores / np.sum(scores, axis=1, keepdims=True)

        return np.argmax(probability, axis=1)

    # fits the model according to inputs
    def fit(self, hidden_nodes, epochs=20000, print_loss=False):

        # initialize randomized weights
        W1 = np.random.normal(0, 1, [self.input_dim, hidden_nodes])
        W2 = np.random.normal(0, 1, [hidden_nodes, self.output_dim])
        b1 = np.zeros((1, hidden_nodes))
        b2 = np.zeros((1, self.output_dim))

        loss = []
        weights = {}

        for i in range(epochs):

            # forward propogation
            z1 = self.X.dot(W1) + b1
            a1 = self.activation(z1)
            z2 = a1.dot(W2) + b2
            scores = self.softmax(z2)

            # back propogation
            delta_3 = scores
            delta_3[range(self.num_samples), self.y] -= 1
            delta_product_w2 = (a1.T).dot(delta_3)
            delta_product_b2 = np.sum(delta_3, axis=0, keepdims=True)
            delta_2 = delta_3.dot(W2.T) * self.activation_derivative(a1)
            delta_product_w1 = np.dot(self.X.T, delta_2)
            delta_product_b1 = np.sum(delta_2, axis=0)

            delta_product_w2 += self.lamda * W2
            delta_product_w1 += self.lamda * W1

            # update weights
            W1 += -self.learning_rate * delta_product_w1
            b1 += -self.learning_rate * delta_product_b1
            W2 += -self.learning_rate * delta_product_w2
            b2 += -self.learning_rate * delta_product_b2

            weights = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
            loss.append(self.calculate_loss(weights))

            if print_loss and i % 500 == 0:
                print("Loss after iteration %i: %f" % (i, loss[i]))

        # To report the final activations
        z1 = self.X.dot(W1) + b1
        a1 = self.activation(z1)
        z2 = a1.dot(W2) + b2
        scores = self.softmax(z2)

        return weights, loss, scores

    # Calculates the accuracy of the current model,
    # prediction vs the actual
    def accuracy(self, weights, X, y):
        same = 0
        flag = 0
        for i in X:
            predict = self.predict(weights, i)[0]
            if predict == y[flag]:
                same += 1
            flag += 1

        return 100.0 * same / len(y)


# Use this to train over multiple values of # of hidden nodes
# to see how no. of hidden nodes affects the accuracy
def main():

    # keep activation as 'tanh', 'sigmoid' or 'relu' to
    # use different activation functions
    set_model = FNN(activation='tanh')
    X_test, y_test = set_model.load_data('test')
    i = 3
    ac = []
    while i < 34:
        weights, loss, scores = set_model.fit(i, print_loss=False)
        acc = set_model.accuracy(weights, X_test, y_test)
        ac.append(acc)
        print('FOR I=' , i , 'Test accuracy: {:0.2f}%'.format(acc))
        i += 3
    plt.plot([x * 3 for x in range(len(ac))], ac)
    plt.show()


# Use this to check the accuracy for a single value of no. of hidden
# nodes. Also prints out the loss vs. no. of iterations
def main2():
    set_model = FNN(activation='tanh')
    X_test, y_test = set_model.load_data('test')
    weights, loss, scores = set_model.fit(5, print_loss=False)
    acc = set_model.accuracy(weights, X_test, y_test)
    print('Test accuracy: {:0.2f}%'.format(acc))
    plt.plot(range(20000), loss)
    plt.show()


if __name__ == '__main__':
    main()