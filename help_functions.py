import numpy as np
import json
import os


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


def vector2list_of_matrices(vector, split_size):
    """
    Reshape binary vector to list of diagonal matrices of size split_size.

    :param vector: binary vector of {-1, 1}
    :param split_size: size of each matrix (len(vector) must be divisible by split_size)
    :return: list of diagonal matrices (matrices are 2D numpy arrays)
    """
    split_vector = np.split(vector, len(vector) / split_size)
    return list(map(lambda x: np.diag(x), split_vector))


def multiply_kernels_with_context(curr_w, vector):
    """
    Multiply all convolutional kernels with context matrices.

    :param curr_w: current model weights
    :param vector: binary vector of {-1, 1}
    :return: new multiplied weights that have the same shape as curr_w
    """
    _, kernel_size, tensor_width, num_of_conv_layers = curr_w.shape

    curr_w_reshaped = np.reshape(curr_w, (tensor_width * num_of_conv_layers, kernel_size, kernel_size))
    context_matrices = vector2list_of_matrices(vector, kernel_size)

    new_w = np.array([mat_weights @ mat_context for mat_weights, mat_context in zip(curr_w_reshaped, context_matrices)])
    return np.reshape(new_w, curr_w.shape)


def zero_out_vector(vec, proportion_0):
    """
    Zero out 'proportion_0' values in vector 'vec' with the lowest absolute magnitude.

    :param vec: vector of numeric values (numpy array)
    :param proportion_0: share of zeros we want in vector 'vec' (value between 0 and 1)
    :return: new vector with specified proportion of 0
    """
    vec_sorted = sorted(np.absolute(vec))
    abs_threshold = vec_sorted[round(len(vec) * proportion_0)]
    mask = (np.absolute(vec) > abs_threshold).astype(float)
    return mask * vec


def get_current_saved_results(filename, additional_labels):
    """
    Load dictionary with results from JSON file and create new keys if necessary.

    :param filename: name of the file this function is called from (without .py extension)
    :param additional_labels: list of strings providing extra information
    :return: dictionary with current data, list of keys for dictionary values
    """
    with open('saved_data/multiple_results.json', 'r') as fp:
        data = json.load(fp)

    # add new keys to the dictionary if possible
    dict_keys = []
    for label in additional_labels:
        curr_filename = filename + '_' + label
        if curr_filename not in data:
            data[curr_filename] = []
        dict_keys.append(curr_filename)

    return data, dict_keys


