"""
Based on article 'THE LOTTERY TICKET HYPOTHESIS: FINDING SPARSE, TRAINABLE NEURAL NETWORKS':
https://arxiv.org/pdf/1803.03635.pdf
"""
import os

from datasets import *
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.engine.saving import load_model
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import Callback, LearningRateScheduler

from help_functions import get_current_saved_results
from plots import *
from math import exp
import numpy as np
import time


class TestPerformanceCallback(Callback):
    """
    Callback class for testing normal model performance at the beginning of every epoch.
    """
    def __init__(self, X_test, y_test, mask):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.mask = mask
        self.accuracies = []

    def on_epoch_begin(self, epoch, logs=None):
        # evaluate only on 1.000 images (10% of all test images) to speed-up
        loss, accuracy = model.evaluate(self.X_test[:1000], self.y_test[:1000], verbose=2)
        self.accuracies.append(accuracy * 100)

    def on_batch_begin(self, batch, logs=None):
        # use mask to zero out certain model weights
        i = 0
        for layer in model.layers:
            if isinstance(layer, Dense):
                new_weights = self.mask[i] * layer.get_weights()[0]
                layer.set_weights([new_weights, layer.get_weights()[1]])
                i += 1


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


def train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, batch_size, validation_share, mask):
    """
    Train and evaluate Keras model.

    :param model: Keras model instance
    :param X_train: train input data
    :param y_train: train output labels
    :param X_test: test input data
    :param y_test: test output labels
    :param num_of_epochs: number of epochs to train the model
    :param batch_size: batch size - number of samples per gradient update
    :param validation_share: share of examples to be used for validation
    :param mask: binary mask to know which weights are pruned
    :return: History object and test accuracies for every training epoch
    """
    test_callback = TestPerformanceCallback(X_test, y_test, mask)
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


def normal_training(model, X_train, y_train, X_test, y_test, num_of_epochs, batch_size, mask):
    """
    Train NN model.

    :param model: Keras model instance
    :param X_train: train input data
    :param y_train: train output labels
    :param X_test: test input data
    :param y_test: test output labels
    :param num_of_epochs: number of epochs to train the model
    :param batch_size: batch size - number of samples per gradient update
    :param mask: binary mask to know which weights are pruned
    :return: list of test accuracies for 10 epochs for each task
    """
    original_accuracies = []

    # first training task - original MNIST images
    history, accuracies = train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, batch_size, 0.1, mask)
    original_accuracies.extend(accuracies)

    val_acc = np.array(history.history['val_accuracy']) * 100
    print('\nValidation accuracies: ', 'first', val_acc)

    return original_accuracies


def get_model_weights(model):
    """
    Get model weights and starting mask.

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


def update_mask(model, mask, curr_pruning_numbers):
    """
    Prune weights with the mask change.

    :param model: current Keras model
    :param mask: binary mask to know which weights are pruned
    :param curr_pruning_numbers: list of numbers of neurons to be marked as inactive in each layer
    :return: updated mask
    """
    updated_mask = []
    i = 0
    for layer in model.layers:
        if isinstance(layer, Dense):
            prune_num = curr_pruning_numbers[i]
            m = mask[i]
            m_0 = int((m.shape[0] * m.shape[1]) - np.sum(m))  # number of zeros in mask (all - number of ones)
            all_0 = prune_num + m_0    # total number of zeros in 'm' after update
            w = layer.get_weights()[0]   # without bias
            w *= m    # to zero out already pruned weights
            w_abs_flattened_sorted = sorted(np.absolute(w).flatten())
            abs_threshold = (w_abs_flattened_sorted[all_0] + w_abs_flattened_sorted[all_0 - 1]) / 2
            updated_mask.append((np.absolute(w) > abs_threshold).astype(float))
            i += 1
    return updated_mask


def iterative_pruning(model, init_weights, mask, n, prune_each_step, X_train, y_train, X_test, y_test, num_of_epochs, batch_size, use_init_weights=True):
    """
    Iterative pruning of model weights with lottery ticket hypothesis using mask in n steps.

    :param model: Keras model trained on the whole network with all weights
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
    :param use_init_weights: True if you want to use initial weights after pruning, False if already trained weights
    :return: list of test accuracies, list of strings presenting percentage of remained weights for each step
    """
    accuracies = []
    remained_weights = [100.0]
    for iteration in range(n):
        curr_weights_active = [np.sum(m) for m in mask]
        curr_pruning_numbers = [int(round(weights_num * prune_each_step)) for weights_num in curr_weights_active]

        # for each dense layer get curr_pruning_number of the lowest weights by magnitude and mark them with 0 in mask
        mask = update_mask(model, mask, curr_pruning_numbers)

        percentage_of_remained_weights = (sum([int(np.sum(m)) for m in mask]) / model.count_params()) * 100
        remained_weights.append(percentage_of_remained_weights)
        print("\n\nIteration %d\nRemaining weights in each layer: [%s] ----- %.1f %% of all starting weights\n\n" %
              (iteration, ', '.join([str(int(np.sum(m))) for m in mask]), percentage_of_remained_weights))

        # element-wise multiply new mask with init_weights and set that as new model weights
        if use_init_weights:
            i = 0
            for layer in model.layers:
                if isinstance(layer, Dense):
                    new_weights = mask[i] * init_weights[i][0]
                    layer.set_weights([new_weights, init_weights[i][1]])
                    i += 1
        else:
            i = 0
            for layer in model.layers:
                if isinstance(layer, Dense):
                    new_weights = mask[i] * layer.get_weights()[0]
                    layer.set_weights([new_weights, layer.get_weights()[1]])
                    i += 1

        # train model with the same parameters as before just with different weights considering mask and save results
        acc = normal_training(model, X_train, y_train, X_test, y_test, num_of_epochs, batch_size, mask)
        accuracies.extend(acc)
    return accuracies, list(map(lambda x: str(round(x, 1)) + ' %', remained_weights))


if __name__ == '__main__':
    input_size = (28, 28)
    num_of_units = 150
    num_of_classes = 10

    num_of_epochs = 10
    batch_size = 600

    X_train, y_train, X_test, y_test = prepare_data(num_of_classes)

    prune_share = 0.99   # share of weights we want to prune in each layer
    n = 9   # number of steps to get to the sparsity wanted
    prune_each_step = prune_share_for_each_step(prune_share, n)

    data, dict_keys = get_current_saved_results(os.path.basename(__file__)[:-3], ['iterative_pruning'])

    plot_multiple_results(dict_keys, [], ['tab:orange'], 'Epoch', 'Accuracy (%)',
                          [i * num_of_epochs for i in range(n + 1)], 5, 100, show_CI=False, text_strings=['100.0 %',
                          '59.9 %', '35.9 %', '21.5 %', '12.9 %', '7.7 %', '4.6 %', '2.8 %', '1.7 %', '1.0 %'])

    num_of_runs = 0
    for i in range(num_of_runs):
        print('\n\n------\nRun #%d\n------\n\n' % (i + 1))

        # iterative pruning without changing weights with initial ones
        model = simple_model(input_size, num_of_units, num_of_classes)
        _, mask = get_model_weights(model)
        acc_train_no_init = normal_training(model, X_train, y_train, X_test, y_test, num_of_epochs, batch_size, mask)
        acc_pruning_no_init, remained_weights = iterative_pruning(model, None, mask, n, prune_each_step, X_train, y_train, X_test,
                                                   y_test, num_of_epochs, batch_size, use_init_weights=False)
        all_accuracies_no_init = acc_train_no_init + acc_pruning_no_init

        data[dict_keys[0]].append(all_accuracies_no_init)

        with open('saved_data/multiple_results.json', 'w') as fp:
            json.dump(data, fp, sort_keys=True, indent=4)

    '''
    # iterative pruning with lottery ticket hypothesis
    model = simple_model(input_size, num_of_units, num_of_classes)
    init_weights, mask = get_model_weights(model)
    acc_train = normal_training(model, X_train, y_train, X_test, y_test, num_of_epochs, batch_size, mask)
    acc_pruning, remained_weights = iterative_pruning(model, init_weights, mask, n, prune_each_step,
                                                      X_train, y_train, X_test, y_test, num_of_epochs, batch_size)
    all_accuracies = acc_train + acc_pruning

    plot_general(all_accuracies, all_accuracies_no_init, ['LT test accuracy', 'no LT test accuracy', '% of remaining weights'],
                 'Lottery ticket iterative pruning with %s weights' % f'{model.count_params():,}', 'epoch', 'accuracy (%)',
                 [i * num_of_epochs for i in range(n + 1)], min(all_accuracies + all_accuracies_no_init) - 2,
                 max(all_accuracies + all_accuracies_no_init) + 2, text_strings=remained_weights)
    '''


