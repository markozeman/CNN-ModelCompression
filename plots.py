import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math


def show_grayscale_image(img):
    """
    Show grayscale image (1 channel).

    :param img: image to plot
    :return: None
    """
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()


def show_image(img):
    """
    Show coloured image (3 channels).

    :param img: image to plot
    :return: None
    """
    plt.imshow(img)
    plt.show()


def plot_fit_history(history):
    """
    Plot changes of loss and accuracy during training.

    :param history: History object
    :return: None
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])

    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'])
        plt.plot(history.history['val_loss'])
        plt.legend(['Accuracy (train)', 'Loss (train)', 'Accuracy (test)', 'Loss (test)'])
    else:
        plt.legend(['Accuracy (train)', 'Loss (train)'])

    plt.title('Model accuracy and loss during training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy / Loss')
    plt.show()


def plot_accuracies_over_time(normal_accuracies, superposition_accuracies):
    """
    Plot accuracies of original test images over time with normal and superposition learning.

    :param normal_accuracies: list of accuracies with normal training
    :param superposition_accuracies: list of accuracies with superposition training
    :return: None
    """
    plt.plot(normal_accuracies)
    plt.plot(superposition_accuracies)
    plt.vlines(9.5, 0, 100, colors='red', linestyles='dotted')
    plt.legend(['Baseline model', 'Superposition model'])
    plt.title('Model accuracy with normal and superposition training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.show()


def plot_general(line_1, line_2, legend_lst, title, x_label, y_label, vertical_lines_x, vl_min, vl_max, text_strings=None):
    """
    Plot two lines on the same plot with additional general information.

    :param line_1: y values of the first line
    :param line_2: y values of the second line
    :param legend_lst: list of two values -> [first line label, second line label]
    :param title: plot title (string)
    :param x_label: label of axis x (string)
    :param y_label: label of axis y (string)
    :param vertical_lines_x: x values of where to draw vertical lines
    :param vl_min: vertical lines minimum y value
    :param vl_max: vertical lines maximum y value
    :param text_strings: optional list of text strings to add to the bottom of vertical lines
    :return: None
    """
    plt.plot(line_1)
    plt.plot(line_2)
    plt.vlines(vertical_lines_x, vl_min, vl_max, colors='red', linestyles='dotted')
    plt.legend(legend_lst)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for i in range(len(text_strings)):
        plt.text(vertical_lines_x[i] + 0.25, vl_min, text_strings[i], color='r')
    plt.show()


def plot_lr(learning_rates):
    """
    Plot changes of learning rate over time.

    :param learning_rates: list of learning rates
    :return: None
    """
    plt.plot(learning_rates)
    plt.title('Change of learning rate over time')
    plt.xlabel('Epoch')
    plt.ylabel('Learning rate')
    plt.show()


def plot_weights_histogram(x, bins):
    """
    Plot weights values on histogram.

    :param x: data/values to plot
    :param bins: number of bins on histogram
    :return: None
    """
    plt.hist(x, bins=bins)
    plt.title('Values of trained weights in the network')
    plt.xlabel('Weight value')
    plt.ylabel('Occurrences')
    plt.show()


def weights_heatmaps(W_matrices):
    """
    Plot heat maps of weights from every layer in the network.

    :param W_matrices: list of 2D numpy arrays which represent weights between layers
    :return: None
    """
    # norm_matrix = (W_matrix - np.min(W_matrix)) / np.ptp(W_matrix)   # normalise matrix between [0, 1]
    plt.figure()
    if len(W_matrices) <= 3:
        plot_layout = (1, len(W_matrices))
    else:
        plot_layout = (2, math.ceil(len(W_matrices) / 2))

    for layer_index, weights_matrix in enumerate(W_matrices):
        plt.subplot(*plot_layout, layer_index + 1)
        sns.heatmap(weights_matrix, cmap='Blues', linewidth=0)
        plt.title("Heatmap of %s weights between layers %d and %d" % (str(weights_matrix.shape), layer_index, layer_index + 1))
    plt.show()


if __name__ == '__main__':
    weights_matrices = [np.array([[-3, -2.5, -1], [1, 5, 8], [1, 2, 8], [1, 7, 12]]), np.array([[3, 7, 8], [1, 7, 8], [5, 5, 1], [0, 1, 5]]),
                        np.array([[-3, -2.5, -1], [1, 5, 8], [1, 2, 8], [1, 7, 12]]), np.array([[3, 7, 8], [1, 7, 8], [5, 5, 1], [0, 1, 5]]),
                        np.array([[-3, -2.5, -1], [1, 5, 8], [1, 2, 8], [1, 7, 12]]), np.array([[3, 7, 8], [1, 7, 8], [5, 5, 1], [0, 1, 5]])]
    weights_heatmaps(weights_matrices)



