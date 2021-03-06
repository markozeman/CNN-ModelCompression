import json
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
    # plt.plot([i * 10 for i in range(len(superposition_accuracies))], superposition_accuracies)
    plt.vlines(10, 0, 100, colors='red', linestyles='dotted')
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
    if text_strings is not None:
        for i in range(len(text_strings)):
            plt.text(vertical_lines_x[i] + 0.25, vl_min, text_strings[i], color='r')
    plt.show()


def plot_multiple_results(dict_keys, legend_lst, colors, x_label, y_label, vertical_lines_x, vl_min, vl_max, show_CI=True, text_strings=None):
    """
    Plot more lines from the saved results on the same plot with additional information.

    :param dict_keys: list of strings that represent keys in saved JSON file
    :param legend_lst: list of label values (length of dict_keys)
    :param colors: list of colors used for lines (length of dict_keys)
    :param x_label: label of axis x (string)
    :param y_label: label of axis y (string)
    :param vertical_lines_x: x values of where to draw vertical lines
    :param vl_min: vertical lines minimum y value
    :param vl_max: vertical lines maximum y value
    :param show_CI: show confidence interval range (boolean)
    :param text_strings: optional list of text strings to add to the bottom of vertical lines
    :return: None
    """
    with open('saved_data/multiple_results.json', 'r') as fp:
        data = json.load(fp)

    font = {'size': 20}
    plt.rc('font', **font)

    # plt.ylim(33, 85)

    # plot lines with confidence intervals
    for i, dict_key in enumerate(dict_keys):
        matrix = np.array(data[dict_key])
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)

        # take only every n-th element of the array
        n = 1
        mean = mean[0::n]
        std = std[0::n]

        # plot the shaded range of the confidence intervals (mean +/- 2*std)
        if show_CI:
            up_limit = mean + (2 * std)
            up_limit[up_limit > 100] = 100  # cut accuracies above 100
            down_limit = mean - (2 * std)
            plt.fill_between(range(0, mean.shape[0] * n, n), up_limit, down_limit, color=colors[i], alpha=0.25)

        # plot the mean on top (every other line is dashed)
        if i % 2 == 0:
            plt.plot(range(0, mean.shape[0] * n, n), mean, colors[i], linewidth=3)
        else:
            plt.plot(range(0, mean.shape[0] * n, n), mean, colors[i], linewidth=3, linestyle='--')

    # # added only for baseline horizontal line
    # plt.plot(range(100), [58]*100, 'tab:orange', linewidth=3, linestyle='--')
    #
    # # added only for ResNet18 results from Superposition article
    # first_5 = [15, 40, 70, 75, 79, 81, 82, 82.5, 82.5, 82] + list(np.linspace(81.8, 74, num=40) + np.random.normal(0, 0.2, 40))
    # last_5 = [first_5[-1]] + list(np.linspace(73.8, 64, num=50))
    # plt.plot(range(50), first_5, 'tab:purple', linewidth=3)
    # plt.plot(range(49, 100), last_5, 'tab:purple', linewidth=3, linestyle='--')

    if legend_lst:
        plt.legend(legend_lst)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.vlines(vertical_lines_x, vl_min, vl_max, colors='k', linestyles='dashed', linewidth=2, alpha=0.5)
    if text_strings is not None:
        for i in range(len(text_strings)):
            plt.text(vertical_lines_x[i] + 0.5, vl_min, text_strings[i], color='k', alpha=0.5)
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



