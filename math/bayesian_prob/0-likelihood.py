#!/usr/bin/env python3
"""Calculate likelihood
"""


import numpy as np


def likelihood(x, n, P):
    """Returns: a 1D numpy.ndarray containing the likelihood of obtaining
        the data, x and n, for each probability in P, respectively
    """
    if type(n) is not int or n <= 0:
        raise ValueError('n must be a positive integer')
    if type(x) is not int or x < 0:
        error = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(error)
    if x > n:
        raise ValueError('x cannot be greater than n')
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    for i in P:
        if i < 0 or i > 1:
            raise ValueError('All values in P must be in the range [0, 1]')
    factorial = np.math.factorial
    combinatory = factorial(n)/(factorial(x)*factorial(n-x))
    likelihood = combinatory * (P**x) * ((1-P)**(n-x))
    return likelihood
