from datasets import *
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import Callback, LearningRateScheduler
from plots import *
from math import exp
import numpy as np


class TestPerformanceCallback(Callback):
    """
    Callback class for testing normal model performance at the beginning of every epoch.
    """
    def __init__(self, X_test, y_test):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.accuracies = []

    def on_epoch_begin(self, epoch, logs=None):
        # evaluate only on 1.000 images (10% of all test images) to speed-up
        loss, accuracy = model.evaluate(self.X_test[:1000], self.y_test[:1000], verbose=2)
        self.accuracies.append(accuracy * 100)


class TestSuperpositionPerformanceCallback(Callback):
    """
    Callback class for testing superposition model performance at the beginning of every epoch.
    """
    def __init__(self, X_test, y_test, context_matrices, model, task_index):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.context_matrices = context_matrices
        self.model = model
        self.task_index = task_index
        self.accuracies = []

    def on_epoch_begin(self, epoch, logs=None):
        if self.task_index == 0:    # first task (original MNIST images) - we did not use context yet
            self.accuracies.append(-1)
            return

        # save current model weights (without bias node)
        curr_w_matrices = []
        curr_bias_vectors = []
        for layer in self.model.layers[1:]:  # first layer is Flatten so we skip it
            curr_w_matrices.append(layer.get_weights()[0])
            curr_bias_vectors.append(layer.get_weights()[1])

        # temporarily change model weights to be suitable for first task (original MNIST images), (without bias node)
        for i, layer in enumerate(self.model.layers[1:]):  # first layer is Flatten so we skip it
            # # multiplying with inverse matrices to 'unfold'
            # context_inverse_multiplied = np.linalg.inv(self.context_matrices[self.task_index][i])
            # for task_i in range(self.task_index - 1, 0, -1):
            #     context_inverse_multiplied = context_inverse_multiplied @ np.linalg.inv(self.context_matrices[task_i][i])


            # not multiplying with inverse because inverse is the same in binary superposition with {-1, 1} on the diagonal
            # using only element-wise multiplication on diagonal vectors for speed-up
            context_inverse_multiplied = np.diagonal(self.context_matrices[self.task_index][i])
            for task_i in range(self.task_index - 1, 0, -1):
                context_inverse_multiplied = np.multiply(context_inverse_multiplied, np.diagonal(self.context_matrices[task_i][i]))
            context_inverse_multiplied = np.diag(context_inverse_multiplied)

            layer.set_weights([curr_w_matrices[i] @ context_inverse_multiplied, curr_bias_vectors[i]])

        # evaluate only on 1.000 images (10% of all test images) to speed-up
        loss, accuracy = self.model.evaluate(self.X_test[:1000], self.y_test[:1000], verbose=2)
        self.accuracies.append(accuracy * 100)

        # change model weights back (without bias node)
        for i, layer in enumerate(self.model.layers[1:]):  # first layer is Flatten so we skip it
            layer.set_weights([curr_w_matrices[i], curr_bias_vectors[i]])


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


def train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, batch_size=32, validation_share=0.0,
                mode='normal', context_matrices=None, task_index=None, saved_weights=None):
    """
    Train and evaluate Keras model.

    :param model: Keras model instance
    :param X_train: train input data
    :param y_train: train output labels
    :param X_test: test input data
    :param y_test: test output labels
    :param num_of_epochs: number of epochs to train the model
    :param batch_size: batch size - number of samples per gradient update (default = 32)
    :param validation_share: share of examples to be used for validation (default = 0)
    :param mode: string for learning mode, important for callbacks - possible values: 'normal', 'superposition'
    :param context_matrices: multidimensional numpy array with random context (binary superposition), only used when mode = 'superposition'
    :param task_index: index of current task, only used when mode = 'superposition'
    :param saved_weights: parameter not used in this file
    :return: History object and 2 lists of test accuracies for every training epoch (normal, superposition)
    """
    test_callback = TestPerformanceCallback(X_test, y_test)
    test_superposition_callback = TestSuperpositionPerformanceCallback(X_test, y_test, context_matrices, model, task_index)
    lr_callback = LearningRateScheduler(lr_scheduler)

    callbacks = [lr_callback]
    if mode == 'normal':
        callbacks.append(test_callback)
    elif mode == 'superposition':
        callbacks.append(test_superposition_callback)

    history = model.fit(X_train, y_train, epochs=num_of_epochs, batch_size=batch_size, verbose=2,
                        validation_split=validation_share, callbacks=callbacks)
    return history, test_callback.accuracies, test_superposition_callback.accuracies


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
        C1 = np.diag(random_binary_vector(num_of_units))
        C2 = np.diag(random_binary_vector(num_of_units))
        C3 = np.diag(random_binary_vector(num_of_classes))
        context_matrices.append([C1, C2, C3])
    return context_matrices


def normal_training(model, X_train, y_train, X_test, y_test, num_of_epochs, num_of_tasks, input_size, num_of_classes, num_of_units, batch_size=32):
    """
    Train model for 'num_of_tasks' tasks, each task is a different permutation of input images.
    Check how accuracy for original images is changing through tasks using normal training.

    :param model: Keras model instance
    :param X_train: train input data
    :param y_train: train output labels
    :param X_test: test input data
    :param y_test: test output labels
    :param num_of_epochs: number of epochs to train the model
    :param num_of_tasks: number of different tasks (permutations of original images)
    :param input_size: image input size in pixels
    :param num_of_classes: number of different classes/output labels
    :param num_of_units: number of neurons in each hidden layer
    :param batch_size: batch size - number of samples per gradient update (default = 32)
    :return: list of test accuracies for 10 epochs for each task
    """
    original_accuracies = []

    # first training task - original MNIST images
    history, accuracies, _ = train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, batch_size, validation_share=0.1)
    original_accuracies.extend(accuracies)

    val_acc = np.array(history.history['val_accuracy']) * 100
    print('\nValidation accuracies: ', 'first', val_acc)

    # other training tasks - permuted MNIST data
    for i in range(num_of_tasks - 1):
        print("\n\n i: %d \n" % i)
        permuted_X_train = permute_images(X_train)
        history, accuracies, _ = train_model(model, permuted_X_train, y_train, X_test, y_test, num_of_epochs, batch_size, validation_share=0.1)
        original_accuracies.extend(accuracies)

        val_acc = np.array(history.history['val_accuracy']) * 100
        print('\nValidation accuracies: ', i, val_acc)

    return original_accuracies


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

        new_w = curr_w @ context_matrices[task_index][i]
        layer.set_weights([new_w, curr_w_bias])


def superposition_training(model, X_train, y_train, X_test, y_test, num_of_epochs, num_of_units, num_of_classes, num_of_tasks, input_size, batch_size=32):
    """
    Train model for 'num_of_tasks' tasks, each task is a different permutation of input images.
    Check how accuracy for original images is changing through tasks using superposition training.

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
    :param batch_size: batch size - number of samples per gradient update (default = 32)
    :return: list of test accuracies for 10 epochs for each task (or validation accuracies for original task)
    """
    original_accuracies = []
    context_matrices = get_context_matrices(num_of_units, num_of_classes, num_of_tasks)

    # multiply random initialized weights with context matrices for each layer (without changing weights from bias node)
    # context_multiplication(model, context_matrices, 0)

    # first training task - original MNIST images
    history, _, _ = train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, batch_size, validation_share=0.1,
                                mode='superposition', context_matrices=context_matrices, task_index=0)

    val_acc = np.array(history.history['val_accuracy']) * 100
    print('\nValidation accuracies: ', 'first', val_acc)

    original_accuracies.extend(val_acc)

    # other training tasks - permuted MNIST data
    for i in range(num_of_tasks - 1):
        print("\n\n i: %d \n" % i)

        # multiply current weights with context matrices for each layer (without changing weights from bias node)
        context_multiplication(model, context_matrices, i + 1)

        permuted_X_train = permute_images(X_train)
        history, _, accuracies = train_model(model, permuted_X_train, y_train, X_test, y_test, num_of_epochs, batch_size, validation_share=0.1,
                                             mode='superposition', context_matrices=context_matrices, task_index=i + 1)
        original_accuracies.extend(accuracies)

        val_acc = np.array(history.history['val_accuracy']) * 100
        print('\nValidation accuracies: ', i, val_acc)

    return original_accuracies


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


if __name__ == '__main__':
    input_size = (28, 28)
    num_of_units = 1000     # not all units/neurons are active
    num_of_classes = 10

    num_of_tasks = 3       # todo - change to 50
    num_of_epochs = 10
    batch_size = 600

    active_neurons_at_start = 100
    neurons_added_each_task = 20
    assert active_neurons_at_start + ((num_of_tasks - 1) * neurons_added_each_task) <= num_of_units

    train_normal = True
    train_superposition = True

    if train_normal:
        X_train, y_train, X_test, y_test = prepare_data(num_of_classes)

        model = simple_model(input_size, num_of_units, num_of_classes)

        acc_normal = normal_training(model, X_train, y_train, X_test, y_test, num_of_epochs, num_of_tasks, input_size,
                                     num_of_classes, num_of_units, batch_size)

        if not train_superposition:
            plot_lr(lr_over_time)
            plot_accuracies_over_time(acc_normal, np.zeros(len(acc_normal)))

    if train_superposition:
        lr_over_time = []  # re-initiate learning rate
        X_train, y_train, X_test, y_test = prepare_data(num_of_classes)

        model = simple_model(input_size, num_of_units, num_of_classes)

        acc_superposition = superposition_training(model, X_train, y_train, X_test, y_test, num_of_epochs, num_of_units,
                                                   num_of_classes, num_of_tasks, input_size, batch_size)

        if not train_normal:
            plot_lr(lr_over_time)
            plot_accuracies_over_time(np.zeros(len(acc_superposition)), acc_superposition)
        else:
            plot_accuracies_over_time(acc_normal, acc_superposition)


