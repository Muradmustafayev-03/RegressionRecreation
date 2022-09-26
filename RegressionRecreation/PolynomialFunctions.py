class Polynomial:
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def eval(self, x):
        n = len(self.coefficients)
        return sum([self.coefficients[i] * x ** i for i in range(n)])

    def derivative(self, x):
        n = len(self.coefficients)
        return sum([self.coefficients[i] * i * x ** (i-1) for i in range(n)])


class LinearFunction(Polynomial):
    def __init__(self, a, b):
        super().__init__([b, a])


class MultivariateFunction:
    def __init__(self, a, coefficients):
        self.a = a
        self.coefficients = coefficients

    def eval(self, x):
        n = len(self.coefficients)
        return self.a + sum([self.coefficients[i] * x[i] for i in range(n)])

    def gradient(self, x):
        return self.coefficients
