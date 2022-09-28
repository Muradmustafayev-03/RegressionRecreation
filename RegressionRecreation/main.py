from Regressors import MultivariateRegressor
import numpy as np

x = np.array([[xi] for xi in range(10)])
y = np.arange(10) * 2 + 4
r = MultivariateRegressor(1)
r.fit(x, y)
print(r.predict(np.array([[xi] for xi in range(10, 15)])))
print(r.get_hypothetical_equation())
print(r.score())
