"""
Based on article 'Superposition of many models into one':
https://arxiv.org/pdf/1902.05522.pdf
"""
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
    Callback class for testing model performance at the beginning of every batch.
    """
    def __init__(self, X_test, y_test):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.accuracies = []

    def on_epoch_begin(self, epoch, logs=None):
        # evaluate only on 1.000 images (10% of all test images) to speed-up
        loss, accuracy = model.evaluate(X_test[:100], y_test[:100], verbose=2)      # todo - change to 1000
        self.accuracies.append(accuracy * 100)


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
        initial_lr = 0.00001
        k = 0.07
        t = len(lr_over_time)
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
    model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, batch_size=32, validation_share=0.0):
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
    :return: History object and a list of test accuracies for every training epoch
    """
    test_callback = TestPerformanceCallback(X_test, y_test)
    lr_callback = LearningRateScheduler(lr_scheduler)
    history = model.fit(X_train, y_train, epochs=num_of_epochs, batch_size=batch_size, verbose=2,
                        validation_split=validation_share, callbacks=[test_callback, lr_callback])
    return history, test_callback.accuracies


def permute_pixels(im):
    """
    Randomly permute pixels of image 'im'.

    :param im: image to be permuted (2D numpy array)
    :return: permuted image (2D numpy array)
    """
    im_1d = im.flatten()
    im_1d_permuted = np.random.permutation(im_1d)
    return np.reshape(im_1d_permuted, im.shape)


def permute_images(images):
    """
    Permute pixels in all images.

    :param images: numpy array of images
    :return: numpy array of permuted images (of the same size)
    """
    return np.array([permute_pixels(im) for im in images])


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
    for _ in range(num_of_tasks):
        C1 = np.diag(random_binary_vector(num_of_units))
        C2 = np.diag(random_binary_vector(num_of_units))
        C3 = np.diag(random_binary_vector(num_of_classes))
        context_matrices.append([C1, C2, C3])
    return context_matrices


def normal_training(model, X_train, y_train, X_test, y_test, num_of_epochs, num_of_tasks, batch_size=32):
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
    :param batch_size: batch size - number of samples per gradient update (default = 32)
    :return: list of test accuracies for 10 epochs for each task
    """
    original_accuracies = []

    # first training task - original MNIST images
    history, accuracies = train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, batch_size, validation_share=0.1)
    original_accuracies.extend(accuracies)

    val_acc = history.history['val_accuracy']
    print('\nValidation accuracies: ', 'first', val_acc)

    # other training tasks - permuted MNIST data
    for i in range(num_of_tasks - 1):
        print("\n\n i: %d \n" % i)
        permuted_X_train = permute_images(X_train)
        history, accuracies = train_model(model, permuted_X_train, y_train, X_test, y_test, num_of_epochs, batch_size, validation_share=0.1)
        original_accuracies.extend(accuracies)

        val_acc = history.history['val_accuracy']
        print('\nValidation accuracies: ', i, val_acc)

    print('original_accuracies', len(original_accuracies), original_accuracies)
    return original_accuracies


def superposition_training(model, X_train, y_train, X_test, y_test, num_of_epochs, num_of_units, num_of_classes, num_of_tasks, batch_size=32):
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
    :param batch_size: batch size - number of samples per gradient update (default = 32)
    :return: list of test accuracies for 10 epochs for each task
    """
    get_context_matrices(num_of_units, num_of_classes, num_of_tasks)


if __name__ == '__main__':
    input_size = (28, 28)
    num_of_units = 1024
    num_of_classes = 10
    model = simple_model(input_size, num_of_units, num_of_classes)

    X_train, y_train, X_test, y_test = get_MNIST()
    y_train = to_categorical(y_train, num_classes=num_of_classes)   # one-hot encode
    y_test = to_categorical(y_test, num_classes=num_of_classes)     # one-hot encode

    num_of_tasks = 15       # todo - change to 50
    num_of_epochs = 10
    batch_size = 600
    acc_normal = normal_training(model, X_train, y_train, X_test, y_test, num_of_epochs, num_of_tasks, batch_size)
    plot_lr(lr_over_time[1:])
    plot_accuracies_over_time(acc_normal, np.zeros(len(acc_normal)))

    # superposition_training(model, X_train, y_train, X_test, y_test, num_of_epochs, num_of_units, num_of_classes, num_of_tasks, batch_size)

