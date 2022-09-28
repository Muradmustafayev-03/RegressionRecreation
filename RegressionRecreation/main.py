from Regressors import MultivariateRegressor, NormalEquation
import numpy as np

x = np.array([[xi] for xi in range(10)])
y = np.arange(10) * 1.5 + 8
w = NormalEquation(np.matrix(x), np.matrix(y))
print(w)
# r = MultivariateRegressor(1)
# r.fit(x, y)
# print(r.predict(np.array([[xi] for xi in range(10, 15)])))
# print(r.get_hypothetical_equation())
# print(r.score())
