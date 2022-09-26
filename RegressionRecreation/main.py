from PolynomialFunctions import *

m = Monomial(2, 3)
print(m, m.eval(2), m.derivative())

p = Polynomial([4, 2, 1, 0, 5])
print(p, p.eval(2), p.derivative())
