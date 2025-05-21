#!/usr/bin/env python3
def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise.
    Args:
        arr1 (list of int/float): First array.
        arr2 (list of int/float): Second array.
    Returns:
        list of int/float: Resultant array after addition.
    """
    # Check if both arrays are of the same length
    if len(arr1) != len(arr2):
        return None
    return [a + b for a, b in zip(arr1, arr2)]
