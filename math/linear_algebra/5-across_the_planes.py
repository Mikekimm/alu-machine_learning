#!/usr/bin/env python3
"""
Add 2D matrices
"""


def add_matrices2D(mat1, mat2):
    """Add the two matrices

    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    sum_array = []
    if len(mat1) != len(mat2):
        return None
    for i in range(len(mat1)):
        if isinstance(mat1[i], list) and len(mat1[i]) != len(mat2[i]):
            return None
        elif not isinstance(mat1[i], list):
            sum_array.append(mat1[i] + mat2[i])
        elif isinstance(mat1[i], list):
            sum_array.append([])
            for j in range(len(mat1[i])):

                sum_array[i].append(mat1[i][j] + mat2[i][j])
    return sum_array


# mat1 = [[1, 2], [3, 4]]
# mat2 = [[5, 6], [7, 8]]
# print(add_matrices2D(mat1, mat2))
# print(mat1)
# print(mat2)
# print(add_matrices2D(mat1, [[1, 2, 3], [4, 5, 6]]))
