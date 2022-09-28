from PolynomialFunctions import Polynomial, MultivariateFunction
from optimisation_algorithms.GradientDescent import BatchGradientDescent
import numpy as np


def MSE(hypothesis, x, y):
    m = len(y)
    return sum([(hypothesis(x[i]) - y[i]) ** 2 for i in range(m)]) / (2 * m)


class Regressor:
    def __init__(self, d: int, polynomial_function, axis):
        self.d = d
        self._X: np.array = None
        self._y: np.array = None
        self.__coefficients: np.array = None
        self.__PolynomialFunction = polynomial_function
        self.__axis = axis

    def get_params(self) -> np.array:
        return self.__coefficients

    def set_params(self, params: np.array):
        self.__coefficients = params
        return self

    def _do_fit(self, mse_gradient, X, y, max_iterations: int = 100000, alpha: float = 0.02, tol: float = 10 ** (-20),
                randomize: bool = False):
        self._X, self._y = X, y
        bgd = BatchGradientDescent(mse_gradient, self.d)
        self.set_params(bgd.optimize(max_iterations, alpha, tol, randomize))
        return self

    def predict(self, X: np.array):
        return np.apply_along_axis(self.__PolynomialFunction(self.get_params()).eval, self.__axis, X)

    def _calculate_score(self, X, y):
        u = sum((y - self.predict(X)) ** 2)
        v = sum((y - sum(y) / len(y)) ** 2)
        return 1 - u/v

    def get_hypothetical_equation(self):
        return self.__PolynomialFunction(self.get_params())


class PolynomialRegressor(Regressor):
    def __init__(self, d: int):
        super().__init__(d, Polynomial, 0)

    def __MSE_gradient(self, coefficients) -> np.array:
        m = len(self._y)
        hypothesis = Polynomial(coefficients).eval
        return \
            np.array([sum([(hypothesis(self._X[i]) - self._y[i]) * self._X[i] ** j for i in range(m)]) / m
                         for j in range(self.d)])

    def fit(self, X: np.array, y: np.array, max_iterations: int = 100000, alpha: float = 0.02, tol: float = 10 ** (-20),
                 randomize: bool = False):
        self._do_fit(self.__MSE_gradient, X, y, max_iterations, alpha, tol, randomize)
        return self

    def score(self):
        return self._calculate_score(self._X, self._y)


class LinearRegressor(PolynomialRegressor):
    def __init__(self):
        super().__init__(2)


class MultivariateRegressor(Regressor):
    def __init__(self, d: int):
        super().__init__(d+1, MultivariateFunction, 1)

    def __MSE_gradient(self, coefficients) -> np.array:
        m = len(self._y)
        hypothesis = MultivariateFunction(coefficients).eval
        return np.array([sum([(hypothesis(self._X[i]) - self._y[i]) for i in range(m)]) / m
                         if j == 0 else
                         sum([(hypothesis(self._X[i]) - self._y[i]) * self._X[i][j-1] for i in range(m)]) / m
                         for j in range(self.d)])

    def fit(self, X: np.array, y: np.array, max_iterations: int = 100000, alpha: float = 0.02, tol: float = 10 ** (-20),
                 randomize: bool = False):
        self._do_fit(self.__MSE_gradient, X, y, max_iterations, alpha, tol, randomize)
        return self

    def score(self):
        return self._calculate_score(self._X, self._y)
