#!/usr/bin/env python3
def add_matrices2D(mat1, mat2):
    """Adds two 2D matrices element-wise.
    Args:
        mat1 (list of list of int/float): First matrix.
        mat2 (list of list of int/float): Second matrix.
    Returns:
        list of list of int/float: Resultant matrix after addition.
    """

    if len(mat1) != len(mat2):
        return None

    if len(mat1[0]) != len(mat2[0]):
        return None

    result = []

    for row1, row2 in zip(mat1, mat2):
        new_row = []
        for a, b in zip(row1, row2):
            new_row.append(a + b)
        result.append(new_row)

    return result
