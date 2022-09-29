from PolynomialFunctions import Polynomial, MultivariateFunction
from optimisation_algorithms.GradientDescent import BatchGradientDescent
import numpy as np


def MSE(hypothesis, x, y):
    m = len(y)
    return sum([(hypothesis(x[i]) - y[i]) ** 2 for i in range(m)]) / (2 * m)


def NormalEquation(X: np.matrix, y: np.matrix):
    X = np.c_[np.ones(len(X)), X]
    return (np.linalg.inv(X.transpose().dot(X)).dot(X.transpose())).dot(y.transpose())


class Regressor:
    """
    Parent class for different kinds of regressors

    Attributes
    ----------
    d: int
        Number of coefficients,
        either the degree of polynomial or the dimension of the input matrix in Multivariate Regression
    axis: int
        Axis on which to apply the function, predefined for each particular subclass

    Methods
    -------
    get_params(self)
        Returns the coefficients of the function after fit

    set_params(self, params: numpy.array)
        Sets coefficients to the function

    _MSE_gradient(self, coefficients)
        Gradient of the Mean Squared Error function, predefined for each particular subclass

    fit(self, X, y, max_iterations: int = 100000, alpha: float = 0.02, tol: float = 10 ** (-20),
            randomize: bool = False)
        Fit the function to the given data

    def predict(self, X: numpy.array)
        Predict using the linear model

    def score(self)
        Return the coefficient of determination of the prediction

    get_hypothetical_equation(self)
        Returns hypothetical equation to estimate the data

    :return:
    """
    def __init__(self, d: int, polynomial_function, axis: int):
        self.d = d + 1
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

    def _MSE_gradient(self, coefficients) -> np.array: pass

    def fit(self, X, y, max_iterations: int = 100000, alpha: float = 0.02, tol: float = 10 ** (-20),
            randomize: bool = False):
        self._X, self._y = X, y
        bgd = BatchGradientDescent(self._MSE_gradient, self.d)
        return self.set_params(bgd.optimize(max_iterations, alpha, tol, randomize))

    def predict(self, X: np.array):
        return np.apply_along_axis(self.__PolynomialFunction(self.get_params()).eval, self.__axis, X)

    def score(self):
        u = sum((self._y - self.predict(self._X)) ** 2)
        v = sum((self._y - sum(self._y) / len(self._y)) ** 2)
        return 1 - u / v

    def get_hypothetical_equation(self):
        return self.__PolynomialFunction(self.get_params())


class PolynomialRegressor(Regressor):
    """
       Attributes
       ----------
       d: int
           Number of coefficients,
           either the degree of polynomial or the dimension of the input matrix in Multivariate Regression

       Methods
       -------
       get_params(self)
           Returns the coefficients of the function after fit

       set_params(self, params: numpy.array)
           Sets coefficients to the function

       _MSE_gradient(self, coefficients)
           Gradient of the Mean Squared Error function

       fit(self, X, y, max_iterations: int = 100000, alpha: float = 0.02, tol: float = 10 ** (-20),
               randomize: bool = False)
           Fit the function to the given data

       def predict(self, X: numpy.array)
           Predict using the linear model

       def score(self)
           Returns the coefficient of determination of the prediction

       get_hypothetical_equation(self)
           Returns hypothetical equation to estimate the data

       :return:
       """
    def __init__(self, d: int):
        super().__init__(d, Polynomial, 0)

    def _MSE_gradient(self, coefficients) -> np.array:
        m = len(self._y)
        hypothesis = Polynomial(coefficients).eval
        return \
            np.array([sum([(hypothesis(self._X[i]) - self._y[i]) * self._X[i] ** j for i in range(m)]) / m
                      for j in range(self.d)])


class LinearRegressor(PolynomialRegressor):
    """
           Attributes
           ----------
           d: int
               Number of coefficients,
               either the degree of polynomial or the dimension of the input matrix in Multivariate Regression

           Methods
           -------
           get_params(self)
               Returns the coefficients of the function after fit

           set_params(self, params: numpy.array)
               Sets coefficients to the function

           _MSE_gradient(self, coefficients)
               Gradient of the Mean Squared Error function

           fit(self, X, y, max_iterations: int = 100000, alpha: float = 0.02, tol: float = 10 ** (-20),
                   randomize: bool = False)
               Fit the function to the given data

           def predict(self, X: numpy.array)
               Predict using the linear model

           def score(self)
               Returns the coefficient of determination of the prediction

           get_hypothetical_equation(self)
               Returns hypothetical equation to estimate the data

           :return:
           """
    def __init__(self):
        super().__init__(1)


class MultivariateRegressor(Regressor):
    """
           Attributes
           ----------
           d: int
               Number of coefficients,
               either the degree of polynomial or the dimension of the input matrix in Multivariate Regression

           Methods
           -------
           get_params(self)
               Returns the coefficients of the function after fit

           set_params(self, params: numpy.array)
               Sets coefficients to the function

           _MSE_gradient(self, coefficients)
               Gradient of the Mean Squared Error function

           fit(self, X, y, max_iterations: int = 100000, alpha: float = 0.02, tol: float = 10 ** (-20),
                   randomize: bool = False)
               Fit the function to the given data

           def predict(self, X: numpy.array)
               Predict using the linear model

           def score(self)
               Returns the coefficient of determination of the prediction

           get_hypothetical_equation(self)
               Returns hypothetical equation to estimate the data

           :return:
           """
    def __init__(self, d: int):
        super().__init__(d, MultivariateFunction, 1)

    def _MSE_gradient(self, coefficients) -> np.array:
        m = len(self._y)
        hypothesis = MultivariateFunction(coefficients).eval
        return np.array([sum([(hypothesis(self._X[i]) - self._y[i]) for i in range(m)]) / m
                         if j == 0 else
                         sum([(hypothesis(self._X[i]) - self._y[i]) * self._X[i][j - 1] for i in range(m)]) / m
                         for j in range(self.d)])

    def fit(self, X: np.array, y: np.array, max_iterations: int = 100000, alpha: float = 0.02, tol: float = 10 ** (-20),
            randomize: bool = False):
        if self.d < 10 ^ 6:
            try:
                self._X, self._y = X, y
                return self.set_params(NormalEquation(X, y))
            except np.linalg.LinAlgError:
                pass
        return super().fit(X, y, max_iterations, alpha, tol, randomize)
