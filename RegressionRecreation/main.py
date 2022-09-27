from Regressors import LinearRegressor
import numpy as np

x = np.arange(10)
y = np.arange(10) * 2 + 4
r = LinearRegressor()
r.fit(x, y)
print(r.get_hypothetical_equation())
print(r.score())
