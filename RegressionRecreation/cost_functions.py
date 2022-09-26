def MSE(data, hypothesis):
    m = len(data)
    return sum([(hypothesis[i] - data[i]) ** 2 for i in range(m)]) / (2 * m)
