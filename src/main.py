from regression.Regressors import MultivariateRegressor
import numpy as np

X = np.random.rand(1000, 4)
y = np.random.rand(1000, 1)

m = MultivariateRegressor(4)
m.fit(X, y, max_iterations=1000)
print(m.get_hypothetical_equation())
