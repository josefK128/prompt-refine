# zero_center_cols.py
# For a given numpy matrix, creates and returns a new matrix with each
# column centered around the column mean, i.e each column element has the
# mean subtracted.
# see 'Understanding Complex Datasets' Skillicorn p. 57

from typing import List
import numpy as np

def action(A: np.array) -> np.array:
    """
    Takes an arbitrary finite matrix A as an argument, calculates the mean of each column,
    and creates a new matrix B where each column is the original column minus the mean.
    """
    means = np.mean(A, axis=0)
    #print(f'\n\nmeans = {means}')
    centered_matrix = A - means
    #print(f'\n\ncentered_matrix = {centered_matrix}')
    return centered_matrix
