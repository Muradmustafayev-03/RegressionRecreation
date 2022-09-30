import numpy as np


def MSE(hypothesis, x, y):
    m = len(y)
    return sum([(hypothesis(x[i]) - y[i]) ** 2 for i in range(m)]) / (2 * m)


def NormalEquation(X: np.matrix, y: np.matrix):
    X = np.c_[np.ones(len(X)), X]
    return (np.linalg.inv(X.transpose().dot(X)).dot(X.transpose())).dot(y.transpose())