"""
Based on article 'Superposition of many models into one':
https://arxiv.org/pdf/1902.05522.pdf
"""
import os
import pickle
import sys

from keras import Model
from keras.engine.saving import load_model

from datasets import *
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import Callback, LearningRateScheduler
from plots import *
from functools import reduce
from math import exp
from help_functions import multiply_kernels_with_context, get_current_saved_results, zero_out_vector
import numpy as np
import tensorflow as tf


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
        self.model = model  # this is only a reference, not a deep copy
        self.task_index = task_index
        self.accuracies = []

    def on_epoch_begin(self, epoch, logs=None):
        if self.task_index == 0:    # first task (original MNIST images) - we did not use context yet
            # evaluate only on 1.000 images (10% of all test images) to speed-up
            loss, accuracy = self.model.evaluate(self.X_test[:1000], self.y_test[:1000], verbose=2)
            self.accuracies.append(accuracy * 100)
            return

        # save current model weights (without bias node)
        curr_w_matrices = []
        curr_bias_vectors = []
        for i, layer in enumerate(self.model.layers):
            curr_w_matrices.append(layer.get_weights()[0])
            curr_bias_vectors.append(layer.get_weights()[1])

        # temporarily change model weights to be suitable for first task (original MNIST images), (without bias node)
        for i, layer in enumerate(self.model.layers):
            context_inverse_multiplied = self.context_matrices[self.task_index][i]
            for task_i in range(self.task_index - 1, 0, -1):
                context_inverse_multiplied = np.multiply(context_inverse_multiplied, self.context_matrices[task_i][i])
            context_inverse_multiplied = np.diag(context_inverse_multiplied)

            layer.set_weights([context_inverse_multiplied @ curr_w_matrices[i], curr_bias_vectors[i]])  # todo: changed here

        # evaluate only on 1.000 images (10% of all test images) to speed-up
        loss, accuracy = self.model.evaluate(self.X_test[:1000], self.y_test[:1000], verbose=2)
        self.accuracies.append(accuracy * 100)

        # change model weights back (without bias node)
        for i, layer in enumerate(self.model.layers):
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


def sum_up_weights(weights_list):
    """
    Element-wise sum up model weights (with biases) by layers.
    Function is suitable only for networks with 3 weights matrices.

    :param weights_list: list of all model weights with biases (len(weights_list) should be >= 2)
    :return: model weights with biases
    """
    return reduce(lambda x, y: [[np.add(x[0][0], y[0][0]), np.add(x[0][1], y[0][1])],
                                [np.add(x[1][0], y[1][0]), np.add(x[1][1], y[1][1])],
                                [np.add(x[2][0], y[2][0]), np.add(x[2][1], y[2][1])]],
            weights_list)


def simple_model(input_size, num_of_classes):
    """
    Create simple CNN model.

    :param input_size: image input size in pixels
    :param num_of_classes: number of different classes/output labels
    :return: Keras model instance
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(*input_size, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def simple_nn(input_size, num_of_classes):
    """
    Create simple NN model with one hidden layer.

    :param input_size: vector input size
    :param num_of_classes: number of different classes/output labels
    :return: Keras model instance
    """
    model = Sequential()
    model.add(Dense(1000, activation='relu', input_shape=(input_size, )))
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
    :param mode: string for learning mode, important for callbacks - possible values: 'normal', 'superposition', 'real superposition'
    :param context_matrices: multidimensional numpy array with random context (binary superposition), only used when mode = 'superposition' or 'real superposition'
    :param task_index: index of current task, only used when mode = 'superposition'
    :param saved_weights: weights of the model at the end of each task, only used when mode = 'real superposition'
    :return: History object and 3 lists of test accuracies for every training epoch (normal, superposition and real superposition)
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


def random_binary_array(size):
    """
    Create an array of 'size' length consisting only of numbers -1 and 1 (approximately 50% each).

    :param size: shape of the created array
    :return: binary numpy array with values -1 or 1
    """
    vec = np.random.uniform(-1, 1, size)
    vec[vec < 0] = -1
    vec[vec >= 0] = 1
    return vec


def get_context_matrices(num_of_tasks):
    """
    Get random context matrices for neural network that uses binary superposition as a context.

    :param num_of_tasks: number of different tasks
    :return: multidimensional numpy array with random context (binary superposition)
    """
    context_matrices = []
    for i in range(num_of_tasks):
        C1 = random_binary_array(9216)   # todo: changed here
        C2 = random_binary_array(1000)   # todo: changed here
        context_matrices.append([C1, C2])
    return context_matrices


def prepare_MNIST_data(num_of_classes):
    """
    Normalize and prepare MNIST data to be ready for CNN input.

    :param num_of_classes: number of output classes/labels
    :return:  X_train, y_train, X_test, y_test
    """
    X_train, y_train, X_test, y_test = get_MNIST()
    y_train = to_categorical(y_train, num_classes=num_of_classes)  # one-hot encode
    y_test = to_categorical(y_test, num_classes=num_of_classes)  # one-hot encode

    # normalize input images to have values between 0 and 1
    X_train = X_train.astype(dtype=np.float32)
    X_test = X_test.astype(dtype=np.float32)
    X_train /= 255
    X_test /= 255

    # reshape to the right dimensions for CNN
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    return X_train, y_train, X_test, y_test


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
    for i, layer in enumerate(model.layers):
        curr_w = layer.get_weights()[0]
        curr_w_bias = layer.get_weights()[1]

        new_w = np.diag(context_matrices[task_index][i]) @ curr_w  # todo: changed here
        layer.set_weights([new_w, curr_w_bias])


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
    original_accuracies = []
    context_matrices = get_context_matrices(num_of_tasks)

    # context_for_conv_layers = np.array(context_matrices)[:, :2]
    # pickle.dump(context_for_conv_layers, open('MNIST_1CNN_data/conv_layers_contexts.pickle', 'wb'))

    # first training task - original MNIST images
    X_train, y_train, X_test, y_test = get_feature_vector_representation(X_train, y_train, X_test, y_test, (28, 28), 0, proportion_0=0.891493)
    history, _, accuracies = train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, batch_size, validation_share=0.1,
                                         mode='superposition', context_matrices=context_matrices, task_index=0)
    original_accuracies.extend(accuracies)

    val_acc = np.array(history.history['val_accuracy']) * 100
    print('\nValidation accuracies: ', 'first', val_acc)

    # other training tasks - permuted MNIST data
    for i in range(num_of_tasks - 1):
        print("\n\n i: %d \n" % i)

        # multiply current weights with context matrices for each layer (without changing weights from bias node)
        context_multiplication(model, context_matrices, i + 1)

        permuted_X_train = permute_images(X_train)
        permuted_X_train, y_train, X_test, y_test = get_feature_vector_representation(permuted_X_train, y_train, X_test, y_test, (28, 28), i + 1, proportion_0=0.891493)
        history, _, accuracies = train_model(model, permuted_X_train, y_train, X_test, y_test, num_of_epochs, batch_size, validation_share=0.1,
                                             mode='superposition', context_matrices=context_matrices, task_index=i + 1)
        original_accuracies.extend(accuracies)

        val_acc = np.array(history.history['val_accuracy']) * 100
        print('\nValidation accuracies: ', i, val_acc)

    return original_accuracies


def prepare_data(X_train, y_train, X_test, y_test, input_size):
    """
    Normalize and prepare data to be ready for CNN input.

    :param X_train: list of numpy arrays of train data
    :param y_train: list of numpy arrays of train labels
    :param X_test: list of numpy arrays of test data
    :param y_test: list of numpy arrays of test labels
    :param input_size: image input shape
    :return: X_train, y_train, X_test, y_test - in the right form
    """
    # normalize input images to have values between 0 and 1
    X_train = np.array(X_train).astype(dtype=np.float32)
    X_test = np.array(X_test).astype(dtype=np.float32)
    X_train /= 255
    X_test /= 255

    # reshape to the right dimensions for CNN
    X_train = X_train.reshape(X_train.shape[0], *input_size, 1)
    X_test = X_test.reshape(X_test.shape[0], *input_size, 1)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test


def get_feature_vector_representation(X_train, y_train, X_test, y_test, input_size, task_index, proportion_0=0.0):
    """

    """
    rnd = np.random.RandomState(42)
    X_train, y_train, X_test, y_test = prepare_data(X_train, y_train, X_test, y_test, input_size)

    cnn_model = load_model('MNIST_1CNN_data/CNN_model.h5')
    conv_contexts = pickle.load(open('MNIST_1CNN_data/conv_layers_contexts.pickle', 'rb'))

    # unfold weights of 'cnn_model' with 'conv_contexts'
    curr_w_matrices = []
    curr_bias_vectors = []
    for layer_index, layer in enumerate(cnn_model.layers):
        if layer_index < 2:  # conv layer
            curr_w_matrices.append(layer.get_weights()[0])
            curr_bias_vectors.append(layer.get_weights()[1])

    for layer_index, layer in enumerate(cnn_model.layers):
        if layer_index < 2:   # conv layer
            context_vector = conv_contexts[49][layer_index]
            for task_i in range(49, task_index, -1):
                context_vector = np.multiply(context_vector, conv_contexts[task_i][layer_index])

            new_w = np.reshape(np.multiply(curr_w_matrices[layer_index].flatten(), context_vector), curr_w_matrices[layer_index].shape)
            layer.set_weights([new_w, curr_bias_vectors[layer_index]])

    # 3 is the index of Flatten layer which outputs feature vector after convolution and pooling
    intermediate_layer_model = Model(inputs=cnn_model.input, outputs=cnn_model.layers[3].output)

    X_train_vectors = (intermediate_layer_model.predict(X_train)).astype(dtype=np.float32)
    X_test_vectors = (intermediate_layer_model.predict(X_test)).astype(dtype=np.float32)

    print('before: ', X_train_vectors.shape, X_test_vectors.shape)

    # make input even more sparse, with 'proportion_0' of zero values
    for index in range(X_train_vectors.shape[0]):
        X_train_vectors[index] = zero_out_vector(X_train_vectors[index], proportion_0)   # only zero out elements

    for index in range(X_test_vectors.shape[0]):
        X_test_vectors[index] = zero_out_vector(X_test_vectors[index], proportion_0)  # only zero out elements
    # plot_weights_histogram(X_train_vectors[0], 30)  # to test new weights distribution

    # from sklearn.random_projection import GaussianRandomProjection
    # transformer = GaussianRandomProjection(n_components=1000, random_state=rnd)
    # XX = (np.vstack((X_train_vectors, X_test_vectors))).astype(dtype=np.float32)
    # XX = transformer.fit_transform(XX)
    # X_train_vectors = XX[:60000, :]
    # X_test_vectors = XX[60000:, :]

    print('after: ', X_train_vectors.shape, X_test_vectors.shape)
    print('task index:', task_index)

    return X_train_vectors, y_train, X_test_vectors, y_test


if __name__ == '__main__':
    # to avoid cuDNN error (https://github.com/tensorflow/tensorflow/issues/24496)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    input_size = (28, 28)
    num_of_units = 1024  # not used
    num_of_classes = 10

    num_of_tasks = 5       # todo - change to 50
    num_of_epochs = 10
    batch_size = 600

    train_normal = False
    train_superposition = True

    data, dict_keys = get_current_saved_results(os.path.basename(__file__)[:-3], ['acc_superposition_1000non0units'])

    # plot_multiple_results(dict_keys, ['Superposition model', 'Baseline model'], ['tab:blue', 'tab:orange'],
    #                       'Epoch', 'Accuracy (%)', [10], 0, 100)

    input_size = 9216

    num_of_runs = 1
    for i in range(num_of_runs):
        print('\n\n------\nRun #%d\n------\n\n' % (i + 1))

        if train_normal:
            lr_over_time = []  # re-initiate learning rate
            X_train, y_train, X_test, y_test = prepare_MNIST_data(num_of_classes)

            model = simple_model(input_size, num_of_classes)

            acc_normal = normal_training(model, X_train, y_train, X_test, y_test, num_of_epochs, num_of_tasks, batch_size)
            # data[dict_keys[1]].append(acc_normal)

            # if not train_superposition:
            #     plot_lr(lr_over_time)
            #     plot_accuracies_over_time(acc_normal, np.zeros(len(acc_normal)))

        if train_superposition:
            lr_over_time = []  # re-initiate learning rate
            X_train, y_train, X_test, y_test = prepare_MNIST_data(num_of_classes)

            model = simple_nn(input_size, num_of_classes)
            print('layers: ', model.layers)

            acc_superposition = superposition_training(model, X_train, y_train, X_test, y_test, num_of_epochs, num_of_units,
                                                       num_of_classes, num_of_tasks, batch_size)
            data[dict_keys[0]].append(acc_superposition)

            # model.save('MNIST_1CNN_data/CNN_model.h5')

            if not train_normal:
                plot_lr(lr_over_time)
                plot_accuracies_over_time(np.zeros(len(acc_superposition)), acc_superposition)
            else:
                plot_accuracies_over_time(acc_normal, acc_superposition)

        # with open('saved_data/multiple_results.json', 'w') as fp:
        #     json.dump(data, fp, sort_keys=True, indent=4)





