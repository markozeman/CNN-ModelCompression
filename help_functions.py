import numpy as np


def shift_matrix_columns(matrix, num_of_columns):
    """
    Shift columns of matrix by 'num_of_columns' and return shifted matrix.

    :param matrix: 2D numpy array
    :param num_of_columns: number of columns that we want to shift to the right
                           (if this number is bigger than number of columns,
                            the matrix is shifted for 'num_of_columns' % len(columns) to the right)
    :return: shifted matrix (2D numpy array)
    """
    return np.roll(matrix, num_of_columns, axis=1)


