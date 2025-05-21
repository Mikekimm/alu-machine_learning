#!/usr/bin/env python3

def matrix_transpose(matrix):
    """
    Transposes a matrix.
    Args:
        matrix (list of list of int): Input matrix.
    Returns:
        list of list of int: Transposed matrix.
    """
    # Get the number of rows and cols
    rows = len(matrix)
    cols = len(matrix[0])

    # Create a new matri to create a new transpose
    transposed = []

    for col in range(cols):
        new_row = []
        for row in range(rows):
            new_row.append(matrix[row][col])
        transposed.append(new_row)
    return transposed
