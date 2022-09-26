class Polynomial:
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def eval(self, x):
        n = len(self.coefficients)
        return sum([self.coefficients[i] * x ^ i for i in range(n)])

    def derivative(self, x):
        n = len(self.coefficients)
        return sum([self.coefficients[i] * i * x ^ (i-1) for i in range(n)])
