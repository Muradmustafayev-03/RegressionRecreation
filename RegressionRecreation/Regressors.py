from PolynomialFunctions import Polynomial, MultivariateFunction
from optimisation.iterative_algorithms.GradientDescent import BatchGradientDescent
import numpy as np


def MSE(hypothesis, x, y):
    m = len(y)
    return sum([(hypothesis(x[i]) - y[i]) ** 2 for i in range(m)]) / (2 * m)


class Regressor:
    def __init__(self, d: int):
        self.d = d
        self.__X: np.array = None
        self.__y: np.array = None
        self.__coefficients: np.array = None

    def get_params(self) -> np.array:
        return self.__coefficients

    def set_params(self, params: np.array):
        self.__coefficients = params
        return self

    def get_hypothetical_equation(self):
        return Polynomial(self.get_params())

    def score(self):
        pass


class PolynomialRegressor(Regressor):
    def __init__(self, d: int):
        super().__init__(d)

    def __MSE_gradient(self, coefficients) -> np.array:
        m = len(self.__y)
        hypothesis = Polynomial(coefficients).eval
        return \
            np.array([sum([(hypothesis(self.__X[i]) - self.__y[i]) * self.__X[i] ** j for i in range(m)]) / m
                         for j in range(self.d)])

    def fit(self, X: np.array, y: np.array, max_iterations: int = 100000, alpha: float = 0.02, tol: float = 10 ** (-20),
                 randomize: bool = False):
        self.__X = X
        self.__y = y

        bgd = BatchGradientDescent(self.__MSE_gradient, self.d)
        self.set_params(bgd.optimize(max_iterations, alpha, tol, randomize))
        return self

    def predict(self, X: np.array):
        return np.apply_along_axis(Polynomial(self.get_params()).eval, 0, X)


class LinearRegressor(PolynomialRegressor):
    def __init__(self):
        super().__init__(2)


class MultivariateRegressor(Regressor):
    def __init__(self, d: int):
        super().__init__(d)

    def __MSE_gradient(self, coefficients) -> np.array:
        m = len(self.__y)
        hypothesis = MultivariateFunction(coefficients).eval
        return np.array([sum([(hypothesis(self.__X[i]) - self.__y[i]) for i in range(m)]) / m
                         if j == 0 else
                         sum([(hypothesis(self.__X[i]) - self.__y[i]) * self.__X[i] for i in range(m)]) / m
                         for j in range(self.d)])

    def fit(self, X: np.array, y: np.array, max_iterations: int = 100000, alpha: float = 0.02, tol: float = 10 ** (-20),
                 randomize: bool = False):
        self.__X = X
        self.__y = y

        bgd = BatchGradientDescent(self.__MSE_gradient, self.d)
        self.set_params(bgd.optimize(max_iterations, alpha, tol, randomize))
        return self

    def predict(self, X: np.array) -> np.array:
        return np.apply_along_axis(MultivariateFunction(self.get_params()).eval, 0, X)
