from PolynomialFunctions import *

m = Monomial(2, 3)
print(m, m.eval(2), m.derivative())

p = Polynomial([4, 2, 0, 6])
print(p, p.eval(2), p.derivative())

l = LinearFunction(2, 3)
print(l, l.eval(2), l.derivative())

mv = MultivariateFunction(2, [2, 4])
print(mv, mv.eval([2, 7]), mv.gradient())
