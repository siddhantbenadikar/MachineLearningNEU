from hw3 import pca
from hw3 import Clustering
import numpy as np

def part7a():

    # 7 part A
    data = pca.load_data("HW3_Data/dataset1.mat", 'Y')
    Y = np.matrix(data.values)

    # (i)
    pca.plotCluster(Y, 0, 1)

    # (ii)
    pca.plotCluster(Y, 1, 2)

    # (iii)
    U, mean, X_pca = pca.pca(Y, 2)
    pca.plotCluster(X_pca, 0, 1)

    # (iv)
    Z = Clustering.k_means(Y, 2, 5)
    output = []
    for key in Z:
        output.append(Z[key])
    U, mean, X_pca = pca.pca(Y, 2)
    Clustering.plot_data(X_pca.T, output)

    # (v)
    Z = Clustering.k_means(X_pca, 2, 5)
    output = []
    for key in Z:
        output.append(Z[key])
    Clustering.plot_data(X_pca.T, output)



def part7b():
    # 7 part B
    data = pca.load_data("HW3_Data/dataset2.mat", 'Y')
    Y = np.matrix(data.values)
    # (i)
    pca.plotCluster(Y, 0, 1)

    # (ii)
    pca.plotCluster(Y, 1, 2)

    # (iii)
    U, mean, X_pca = pca.pca(Y, 2)
    pca.plotCluster(X_pca, 0, 1)

    # (iv)
    Z = Clustering.k_means(X_pca, 2, 5)
    output = []
    for key in Z:
        output.append(Z[key])
    Clustering.plot_data(X_pca.T, output)

    # (v)
    K = Clustering.gaussian_kernel(Y, 0.3)
    X_kpca = pca.kernel_pca(K, 2)
    Z = Clustering.k_means(X_pca, 2, 5)
    output = []
    for key in Z:
        output.append(Z[key])
    Clustering.plot_data(X_kpca.T, output)

    # (vi)
    # Spectral Clustering
    W = Clustering.generate_W(Y, 5, 0.4)
    Z_s = Clustering.spectral(W, 2)
    output = []
    for key in Z_s:
        output.append(Z_s[key])
    Clustering.plot_data(X_pca.T, output)




def main():
    part7b()


if __name__ == '__main__':
    main()





