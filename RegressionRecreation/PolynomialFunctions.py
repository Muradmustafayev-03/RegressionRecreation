class Monomial:
    def __init__(self, coefficient: float, power: int):
        self.coefficient = coefficient
        self.power = power

    def __str__(self):
        if self.coefficient == 0:
            return '0'
        if self.power == 0:
            return str(self.coefficient)

        coefficient = '' if self.coefficient == 1 else str(self.coefficient)
        power = '' if self.power == 1 else '^' + str(self.power)

        return coefficient + 'x' + power

    def eval(self, x: float):
        return self.coefficient * x ** self.power

    def derivative(self):
        return Monomial(self.coefficient * self.power, self.power - 1)


class Polynomial:
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def __str__(self):
        n = len(self.coefficients)
        return ' + '.join([str(Monomial(self.coefficients[i], i)) for i in range(n)[::-1] if self.coefficients[i] != 0])

    def eval(self, x: float):
        n = len(self.coefficients)
        return sum([Monomial(self.coefficients[i], i).eval(x) for i in range(n)])

    def derivative(self):
        n = len(self.coefficients)
        return Polynomial([self.coefficients[i] * i for i in range(n)][1:])


class LinearFunction(Polynomial):
    def __init__(self, a: float, b: float):
        super().__init__([b, a])


class MultivariateFunction:
    def __init__(self, a, coefficients):
        self.a = a
        self.coefficients = coefficients

    def __str__(self):
        n = len(self.coefficients)
        return str(self.a) + ' + ' + ' + '.join([f'{self.coefficients[i]} * x_{i}' for i in range(n)])

    def eval(self, x):
        n = len(self.coefficients)
        return self.a + sum([self.coefficients[i] * x[i] for i in range(n)])

    def gradient(self):
        return self.coefficients
