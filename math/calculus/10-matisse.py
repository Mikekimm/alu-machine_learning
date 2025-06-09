#!/usr/bin/env python3
""" The derivative of polynomial
"""


def poly_derivative(poly):
    """ Find the derivative of polynomial
    Example: if f(x) = x^3 +4x^2 + 9x +5, poly is equal to [5, 3, 4, 1]

    Args:
        poly (_type_): _description_
    """
    derivative = []
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    for i in range(len(poly)-1, 0, -1):

        derivative.append(poly[i]*i)
    return derivative[::-1]


# print(poly_derivative([]))
