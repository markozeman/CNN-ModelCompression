import copy
import os
from datasets import *
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import Callback, LearningRateScheduler
from help_functions import get_current_saved_results
from plots import *
from math import exp
import numpy as np


def get_test_acc_for_individual_tasks(final_model, context_matrices, test_data, active_neurons_at_start, neurons_added_each_task,
                                      input_size, num_of_units, num_of_classes):
    """
    Use superimposed 'final_model' to test the accuracy for each individual task when weights of the 'final_model'
    are unfolded according to 'context_matrices' and tested on appropriate 'test_data' with
    taking into account masking of weights.

    :param final_model: final model after superposition of all (50) tasks
    :param context_matrices: multidimensional numpy array with random context (binary superposition)
    :param test_data: [(X_test_1, y_test_1), (X_test_2, y_test_2), ...]
    :param active_neurons_at_start: number of active neurons in both hidden layers at the first task
    :param neurons_added_each_task: number of newly activated neurons in both hidden layers for each new task
    :param input_size: image input size in pixels
    :param num_of_units: number of neurons in each hidden layer
    :param num_of_classes: number of different classes/output labels
    :return: list of superposition accuracies for individual tasks
    """
    reverse_accuracies = []
    num_of_tasks = len(test_data)
    for i in range(num_of_tasks - 1, -1, -1):   # go from the last task to the first one
        # save current model weights
        curr_w_matrices = []
        curr_bias_vectors = []
        for layer in final_model.layers[1:]:  # first layer is Flatten so we skip it
            curr_w_matrices.append(layer.get_weights()[0])
            curr_bias_vectors.append(layer.get_weights()[1])

        # mask weights in all layers
        curr_active_neurons = active_neurons_at_start + (i * neurons_added_each_task)
        mask = get_mask(input_size, num_of_units, num_of_classes, curr_active_neurons)

        for layer_index, layer in enumerate(final_model.layers[1:]):  # first layer is Flatten so we skip it
            new_weights = mask[layer_index] * curr_w_matrices[layer_index]
            layer.set_weights([new_weights, curr_bias_vectors[layer_index]])

        # test i-th task accuracy
        loss, accuracy = final_model.evaluate(test_data[i][0], test_data[i][1], verbose=2)
        reverse_accuracies.append(accuracy * 100)

        # reset weights to the ones before masking
        # for layer_index, layer in enumerate(final_model.layers[1:]):  # first layer is Flatten so we skip it
        #     layer.set_weights([curr_w_matrices[layer_index], curr_bias_vectors[layer_index]])

        # unfold weights one task back
        for layer_index, layer in enumerate(final_model.layers[1:]):  # first layer is Flatten so we skip it
            context_inverse_multiplied = np.linalg.inv(np.diag(context_matrices[i][layer_index]))    # matrix inverse is not necessary for binary context
            layer.set_weights([context_inverse_multiplied @ curr_w_matrices[layer_index], curr_bias_vectors[layer_index]])

    return list(reversed(reverse_accuracies))


class ApplyMaskCallback(Callback):
    """
    Callback class for applying mask multiplication to zero out inactive neurons.
    """
    def __init__(self, model, mask):
        super().__init__()
        self.model = model  # this is only a reference, not a deep copy
        self.mask = mask

    def on_batch_begin(self, batch, logs=None):
        # use mask to zero out certain model weights
        for layer_index, layer in enumerate(self.model.layers[1:]):  # first layer is Flatten so we skip it
            new_weights = self.mask[layer_index] * layer.get_weights()[0]
            layer.set_weights([new_weights, layer.get_weights()[1]])


lr_over_time = []   # global variable to store changing learning rates


def lr_scheduler(epoch, lr):
    """
    Learning rate scheduler function to set how learning rate changes each epoch.

    :param epoch: current epoch number
    :param lr: current learning rate
    :return: new learning rate
    """
    global lr_over_time
    lr_over_time.append(lr)
    decay_type = 'exponential'   # 'linear' or 'exponential'
    if decay_type == 'linear':
        lr -= 10 ** -5
    elif decay_type == 'exponential':
        initial_lr = 0.0001
        k = 0.07
        t = len(lr_over_time) % num_of_epochs      # to start each new task with the same learning rate as the first one
        lr = initial_lr * exp(-k * t)
    return max(lr, 0.000001)    # don't let learning rate go to 0


def simple_model(input_size, num_of_units, num_of_classes):
    """
    Create simple NN model with two hidden layers, each has 'num_of_units' neurons.

    :param input_size: image input size in pixels
    :param num_of_units: number of neurons in each hidden layer
    :param num_of_classes: number of different classes/output labels
    :return: Keras model instance
    """
    model = Sequential()
    model.add(Flatten(input_shape=input_size))
    model.add(Dense(num_of_units, activation='relu'))
    model.add(Dense(num_of_units, activation='relu'))
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def train_model(model, X_train, y_train, num_of_epochs, mask, batch_size=32, validation_share=0.0):
    """
    Train and evaluate Keras model.

    :param model: Keras model instance
    :param X_train: train input data
    :param y_train: train output labels
    :param num_of_epochs: number of epochs to train the model
    :param mask: list of binary 2D numpy arrays (1 - active weight, 0 - non active weight)
    :param batch_size: batch size - number of samples per gradient update (default = 32)
    :param validation_share: share of examples to be used for validation (default = 0)
    :return: History object and 2 lists of test accuracies for every training epoch (normal, superposition)
    """
    mask_callback = ApplyMaskCallback(model, mask)
    lr_callback = LearningRateScheduler(lr_scheduler)
    callbacks = [lr_callback, mask_callback]
    history = model.fit(X_train, y_train, epochs=num_of_epochs, batch_size=batch_size, verbose=2,
                        validation_split=validation_share, callbacks=callbacks)
    return history


def prepare_data(num_of_classes):
    """
    Normalize and prepare MNIST data to be ready for NN input.

    :param num_of_classes: number of output classes/labels
    :return:  X_train, y_train, X_test, y_test
    """
    X_train, y_train, X_test, y_test = get_MNIST()
    y_train = to_categorical(y_train, num_classes=num_of_classes)  # one-hot encode
    y_test = to_categorical(y_test, num_classes=num_of_classes)  # one-hot encode

    # normalize input images to have values between 0 and 1
    X_train = X_train.astype(dtype=np.float64)
    X_test = X_test.astype(dtype=np.float64)
    X_train /= 255
    X_test /= 255

    return X_train, y_train, X_test, y_test


def permute_pixels(im, seed):
    """
    Randomly permute pixels of image 'im'.

    :param im: image to be permuted (2D numpy array)
    :param seed: number that serves to have the same permutation for all images in the array
    :return: permuted image (2D numpy array)
    """
    im_1d = im.flatten()
    im_1d_permuted = np.random.RandomState(seed=seed).permutation(im_1d)
    return np.reshape(im_1d_permuted, im.shape)


def permute_images(images):
    """
    Permute pixels in all images.

    :param images: numpy array of images
    :return: numpy array of permuted images (of the same size)
    """
    seed = np.random.randint(low=4294967295, dtype=np.uint32)    # make a random seed for all images in an array
    return np.array([permute_pixels(im, seed) for im in images])


def random_binary_vector(size):
    """
    Create a vector of 'size' length consisting only of numbers -1 and 1 (approximately 50% each).

    :param size: length of the created vector
    :return: binary numpy vector with values -1 or 1
    """
    vec = np.random.uniform(-1, 1, size)
    vec[vec < 0] = -1
    vec[vec >= 0] = 1
    return vec


def get_context_matrices(num_of_units, num_of_classes, num_of_tasks):
    """
    Get random context matrices for neural network that uses binary superposition as a context.

    :param num_of_units: number of neurons in each hidden layer
    :param num_of_classes: number of different classes/output labels
    :param num_of_tasks: number of different tasks (permutations of original images)
    :return: multidimensional numpy array with random context (binary superposition)
    """
    context_matrices = []
    for i in range(num_of_tasks):
        C1 = random_binary_vector(784)
        C2 = random_binary_vector(num_of_units)
        C3 = random_binary_vector(num_of_units)
        context_matrices.append([C1, C2, C3])
    return context_matrices


def context_multiplication(model, context_matrices, task_index):
    """
    Multiply current model weights with context matrices in each layer (without changing weights from bias node).

    :param model: Keras model instance
    :param context_matrices: multidimensional numpy array with random context (binary superposition)
    :param task_index: index of a task to know which context_matrices row to use
    :return: None (but model weights are changed)
    """
    for i, layer in enumerate(model.layers[1:]):  # first layer is Flatten so we skip it
        curr_w = layer.get_weights()[0]
        curr_w_bias = layer.get_weights()[1]

        new_w = np.diag(context_matrices[task_index][i]) @ curr_w
        layer.set_weights([new_w, curr_w_bias])


def superposition_training(model, X_train, y_train, X_test, y_test, num_of_epochs, num_of_units, num_of_classes,
                           num_of_tasks, input_size, batch_size, active_neurons_at_start, neurons_added_each_task):
    """
    Train model for 'num_of_tasks' tasks, each task is a different permutation of input images.
    Check how accuracy is changing through tasks using superposition training.

    :param model: Keras model instance
    :param X_train: train input data
    :param y_train: train output labels
    :param X_test: test input data
    :param y_test: test output labels
    :param num_of_epochs: number of epochs to train the model
    :param num_of_units: number of neurons in each hidden layer
    :param num_of_classes: number of different classes/output labels
    :param num_of_tasks: number of different tasks (permutations of original images)
    :param input_size: image input size in pixels
    :param batch_size: batch size - number of samples per gradient update
    :param active_neurons_at_start: number of active neurons in both hidden layers at the first task
    :param neurons_added_each_task: number of newly activated neurons in both hidden layers for each new task
    :return: final model, context_matrices, test_data for all tasks
    """
    test_data_50tasks = [(X_test, y_test)]
    context_matrices = get_context_matrices(num_of_units, num_of_classes, num_of_tasks)

    # multiply random initialized weights with context matrices for each layer (without changing weights from bias node)
    # context_multiplication(model, context_matrices, 0)

    curr_active_neurons = active_neurons_at_start
    mask = get_mask(input_size, num_of_units, num_of_classes, curr_active_neurons)

    # first training task - original MNIST images
    history = train_model(model, X_train, y_train, num_of_epochs, mask, batch_size, validation_share=0.1)

    val_acc = np.array(history.history['val_accuracy']) * 100
    print('\nValidation accuracies: ', 'first', val_acc)

    # other training tasks - permuted MNIST data
    for i in range(num_of_tasks - 1):
        print("\n\n i: %d \n" % i)

        curr_active_neurons += neurons_added_each_task
        mask = get_mask(input_size, num_of_units, num_of_classes, curr_active_neurons)

        unzero_weights(model, input_size, num_of_classes, active_neurons_at_start, neurons_added_each_task, curr_active_neurons)

        # multiply current weights with context matrices for each layer (without changing weights from bias node)
        context_multiplication(model, context_matrices, i + 1)

        XX = np.vstack((X_train, X_test))
        permuted_XX = permute_images(XX)
        permuted_X_train = permuted_XX[:60000, :]
        permuted_X_test = permuted_XX[60000:, :]

        test_data_50tasks.append((permuted_X_test, y_test))   # save test set for the testing later

        history = train_model(model, permuted_X_train, y_train, num_of_epochs, mask, batch_size, validation_share=0.1)

        val_acc = np.array(history.history['val_accuracy']) * 100
        print('\nValidation accuracies: ', i, val_acc)

        '''
        for layer in model.layers[2:]:
            curr_w_matrices = layer.get_weights()[0]
            magnitude_of_layer_outputs = np.sum(curr_w_matrices, axis=1)
            print('Len:', len(magnitude_of_layer_outputs))
            for i, m in enumerate(magnitude_of_layer_outputs):
                if i == curr_active_neurons:
                    print('----------')
                print(m)
            print('\n\n\n')
        print('\n\n\n\n\n\n')
        '''

    return model, context_matrices, test_data_50tasks


def get_mask(input_size, num_of_units, num_of_classes, curr_active_neurons):
    """
    Make mask list representing active neurons in simple NN model with 2 hidden layers.

    :param input_size: image input size in pixels
    :param num_of_units: number of neurons in each hidden layer
    :param num_of_classes: number of different classes/output labels
    :param curr_active_neurons: number of currently activated neurons in hidden layers
    :return: list of binary 2D numpy arrays (1 - active weight, 0 - non active weight)
    """
    # make mask the right size with all zeros
    mask = [np.zeros((np.prod(input_size), num_of_units)), np.zeros((num_of_units, num_of_units)), np.zeros((num_of_units, num_of_classes))]
    # add ones to mask to represent currently active neurons
    mask[0][:, :curr_active_neurons] = 1
    mask[1][:curr_active_neurons, :curr_active_neurons] = 1
    mask[2][:curr_active_neurons, :] = 1
    return mask


def unzero_weights(model, input_size, num_of_classes, active_neurons_at_start, neurons_added_each_task, curr_active_neurons):
    """
    Un-zero weights to small random values in places where new neurons become active.

    :param model: Keras model instance
    :param input_size: image input size in pixels
    :param num_of_classes: number of different classes/output labels
    :param active_neurons_at_start: number of active neurons in both hidden layers at the first task
    :param neurons_added_each_task: number of newly activated neurons in both hidden layers for each new task
    :param curr_active_neurons: number of currently active neurons in both hidden layers
    :return: None (but model weights are changed)
    """
    for layer_index, layer in enumerate(model.layers[1:]):  # first layer is Flatten so we skip it
        curr_w_matrix = layer.get_weights()[0]

        # make random weights sub-matrix with values between -0.001 and 0.001
        if layer_index == 0:
            submatrix = np.random.uniform(low=-0.001, high=0.001, size=(input_size[0] * input_size[1], neurons_added_each_task))
            curr_w_matrix[:, (curr_active_neurons - neurons_added_each_task): curr_active_neurons] = submatrix
        elif layer_index == 1:
            saved_small_matrix = copy.deepcopy(curr_w_matrix[:active_neurons_at_start, :active_neurons_at_start])
            submatrix = np.random.uniform(low=-0.001, high=0.001, size=(curr_active_neurons, curr_active_neurons))
            curr_w_matrix[:curr_active_neurons, :curr_active_neurons] = submatrix
            curr_w_matrix[:active_neurons_at_start, :active_neurons_at_start] = saved_small_matrix
        elif layer_index == 2:
            submatrix = np.random.uniform(low=-0.001, high=0.001, size=(neurons_added_each_task, num_of_classes))
            curr_w_matrix[(curr_active_neurons - neurons_added_each_task): curr_active_neurons, :] = submatrix

        layer.set_weights([curr_w_matrix, layer.get_weights()[1]])


if __name__ == '__main__':
    input_size = (28, 28)
    num_of_units = 1000     # not all units/neurons are active
    num_of_classes = 10

    num_of_tasks = 5
    num_of_epochs = 10
    batch_size = 600

    active_neurons_at_start = 200
    neurons_added_each_task = 200
    assert active_neurons_at_start + ((num_of_tasks - 1) * neurons_added_each_task) <= num_of_units

    train_superposition = True

    # data, dict_keys = get_current_saved_results(os.path.basename(__file__)[:-3], ['acc_superposition_15tasks_start150_increase0', 'acc_superposition_15tasks_start150_increase17'])
    #
    # plot_multiple_results(dict_keys, ['Superposition model fixed', 'Superposition model increasing'],
    #                       ['tab:orange', 'tab:blue'], 'Epoch', 'Accuracy (%)',
    #                       [], 70, 100, show_CI=False)

    num_of_runs = 1
    for i in range(num_of_runs):
        print('\n\n------\nRun #%d\n------\n\n' % (i + 1))

        if train_superposition:
            lr_over_time = []  # re-initiate learning rate
            X_train, y_train, X_test, y_test = prepare_data(num_of_classes)

            model = simple_model(input_size, num_of_units, num_of_classes)

            final_model, context_matrices, test_data_50tasks = \
                    superposition_training(model, X_train, y_train, X_test, y_test, num_of_epochs, num_of_units,
                                           num_of_classes, num_of_tasks, input_size, batch_size,
                                           active_neurons_at_start, neurons_added_each_task)

            acc_superposition = get_test_acc_for_individual_tasks(final_model, context_matrices, test_data_50tasks, active_neurons_at_start,
                                                                  neurons_added_each_task, input_size, num_of_units, num_of_classes)
            # data[dict_keys[0]].append(acc_superposition)

            # plot_lr(lr_over_time)
            plot_accuracies_over_time(np.zeros(num_of_tasks * num_of_epochs), acc_superposition)

        # with open('saved_data/multiple_results.json', 'w') as fp:
        #     json.dump(data, fp, sort_keys=True, indent=4)

