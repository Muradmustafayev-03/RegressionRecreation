from Regressors import PolynomialRegressor, LinearRegressor
import numpy as np

X0 = np.arange(10)
X1 = np.arange(10, 15)

y = 2 * X0 + 5
y1 = 2 * X1 + 5

r = LinearRegressor()
r.fit(X0, y)
p = r.predict(X1)

print(y1)
print(p)
print(r.get_hypothetical_equation())
