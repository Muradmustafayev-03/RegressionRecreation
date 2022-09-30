from regression.Regressors import PolynomialRegressor
import numpy as np

x = np.arange(10)
y = np.arange(10) * 1.5 + 8
r = PolynomialRegressor(1)
r.fit(x, y)
print(r.predict(np.arange(10, 15)))
print(r.get_hypothetical_equation())
print(r.score(x, y))
