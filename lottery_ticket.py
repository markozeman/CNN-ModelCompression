"""
Based on article 'THE LOTTERY TICKET HYPOTHESIS: FINDING SPARSE, TRAINABLE NEURAL NETWORKS':
https://arxiv.org/pdf/1803.03635.pdf
"""
from datasets import *
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.engine.saving import load_model
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import Callback, LearningRateScheduler
from plots import *
from math import exp
import numpy as np
import time


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

    def on_batch_begin(self, batch, logs=None):
        pass
        # new_w = model.layers[2].get_weights()[0] * mask
        #
        # model.layers[2].set_weights([new_w, model.layers[2].get_weights()[1]])
        #
        # print('begin')
        # print(model.layers[2].get_weights())
        # print()



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
    :return: History object and test accuracies for every training epoch
    """
    test_callback = TestPerformanceCallback(X_test, y_test)
    lr_callback = LearningRateScheduler(lr_scheduler)
    callbacks = [lr_callback, test_callback]

    history = model.fit(X_train, y_train, epochs=num_of_epochs, batch_size=batch_size, verbose=2,
                        validation_split=validation_share, callbacks=callbacks)
    return history, test_callback.accuracies


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


def normal_training(model, X_train, y_train, X_test, y_test, num_of_epochs, batch_size=32):
    """
    Train NN model.

    :param model: Keras model instance
    :param X_train: train input data
    :param y_train: train output labels
    :param X_test: test input data
    :param y_test: test output labels
    :param num_of_epochs: number of epochs to train the model
    :param batch_size: batch size - number of samples per gradient update (default = 32)
    :return: list of test accuracies for 10 epochs for each task
    """
    original_accuracies = []

    # first training task - original MNIST images
    history, accuracies = train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, batch_size, validation_share=0.1)
    original_accuracies.extend(accuracies)

    val_acc = np.array(history.history['val_accuracy']) * 100
    print('\nValidation accuracies: ', 'first', val_acc)

    return original_accuracies


def get_model_weights(model):
    """
    Get model weights.

    :param model: Keras model instance
    :return: current model weights, mask of the same shape except for bias connections
    """
    model_weights = [layer.get_weights() for layer in model.layers if isinstance(layer, Dense)]
    mask = [np.ones(w[0].shape) for w in model_weights]
    return model_weights, mask


def prune_share_for_each_step(prune_share, n):
    """
    Calculate pruning share for each of n steps.

    :param prune_share: share of weights we want to prune in each layer
    :param n: steps to get to the final pruned network
    :return: prune share for each individual step
    """
    return 1 - ((1 - prune_share) ** (1 / n))


def iterative_pruning(model, init_weights, mask, n, prune_each_step, X_train, y_train, X_test, y_test, num_of_epochs, batch_size):
    """
    Iterative pruning of model weights with lottery ticket hypothesis using mask in n steps.

    :param model: model trained on the whole network with all weights
    :param init_weights: initial model weights before training started
    :param mask: binary mask to know which weights are pruned
    :param n: number of steps to get to the sparsity wanted
    :param prune_each_step: prune share for each individual step
    :param X_train: train input data
    :param y_train: train output labels
    :param X_test: test input data
    :param y_test: test output labels
    :param num_of_epochs: number of epochs to train the model
    :param batch_size: batch size - number of samples per gradient update
    :return: None
    """
    for _ in range(n):
        curr_weights_active = [np.sum(m) for m in mask]
        curr_pruning_numbers = [weights_num * prune_each_step for weights_num in curr_weights_active]

        # for each dense layer get curr_pruning_number of the lowest weights by magnitude and mark them with 0 in mask


        # element-wise multiply new mask with init_weights and set that as new model weights


        # train model with the same parameters as before just with different weights considering mask


        # save test accuracy during learning epochs





if __name__ == '__main__':
    input_size = (28, 28)
    num_of_units = 1000
    num_of_classes = 10

    num_of_epochs = 10
    batch_size = 600

    X_train, y_train, X_test, y_test = prepare_data(num_of_classes)

    model = simple_model(input_size, num_of_units, num_of_classes)

    init_weights, mask = get_model_weights(model)
    print(len(init_weights))
    print(len(mask), np.sum(mask[0]), np.sum(mask[1]), np.sum(mask[2]))

    acc_train = normal_training(model, X_train, y_train, X_test, y_test, num_of_epochs, batch_size)

    plot_general(acc_train, np.zeros(len(acc_train)), ['test accuracy', ''], 'Original MNIST NN learning',
                 'epoch', 'accuracy (%)', [], 0, 0)

    prune_share = 0.99
    n = 5
    prune_each_step = prune_share_for_each_step(prune_share, n)

    iterative_pruning(model, init_weights, mask, n, prune_each_step,
                      X_train, y_train, X_test, y_test, num_of_epochs, batch_size)
