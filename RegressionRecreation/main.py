from Regressors import PolynomialRegressor, MultivariateRegressor, NormalEquation
import numpy as np

x = np.arange(10)
y = np.arange(10) * 1.5 + 8
# w = NormalEquation(np.matrix(x), np.matrix(y))
# print(w)
r = PolynomialRegressor(1)
r.fit(x, y)
print(r.predict(np.arange(10, 15)))
print(r.get_hypothetical_equation())
print(r.score(x, y))
