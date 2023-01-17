import numpy as np


# define target function for training data
def fy(x, m, b, size):
    rnd = 10 * np.random.randn(size)
    labels = np.array([1 if i > 0 else 0 for i in rnd])
    return m * x + b + rnd, labels


# EOF