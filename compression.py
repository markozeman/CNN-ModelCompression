import tensorflow as tf
import numpy as np
import time
from keras import Sequential
from keras.engine.saving import load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adadelta
from kerassurgeon.operations import delete_layer, delete_channels
from superposition import prepare_data
from plots import plot_weights_histogram


def cnn_model(input_size, num_of_classes):
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
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_of_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])
    model.summary()
    return model


def train_model(model, X_train, y_train, num_of_epochs, batch_size=32, validation_share=0.0):
    """
    Train simple NN model with 2 hidden layers on MNIST dataset.

    :param model: Keras model instance
    :param X_train: train input data
    :param y_train: train output labels
    :param num_of_epochs: number of epochs to train the model
    :param batch_size: batch size - number of samples per gradient update (default = 32)
    :param validation_share: share of examples to be used for validation (default = 0)
    :return: History object
    """
    history = model.fit(X_train, y_train, epochs=num_of_epochs, batch_size=batch_size,
                        validation_split=validation_share, verbose=2)
    return history


def calculate_threshold(model, pruning_share):
    """
    Calculate absolute threshold used for weight pruning.

    :param model: Keras model instance
    :param pruning_share: value between 0 and 1 that defines how many weights we want to prune
    :return: threshold
    """
    all_weights = []
    for lyr in model.layers:
        w = lyr.get_weights()
        if w:
            print(lyr)
            print(len(w), w[0].shape)
            all_weights.extend(w[0].flatten())
            all_weights.extend(w[1].flatten())

    all_weights = list(map(lambda x: abs(x), all_weights))    # absolute value of all weights
    all_weights.sort()

    # plot_weights_histogram(all_weights, 100)

    cut_index = round(len(all_weights) * pruning_share)
    threshold = all_weights[cut_index]
    return threshold


def best_channels2prune(dense_layer, num_of_neurons):
    """
    Find indices of the neurons that has the smallest std of Keras Dense layer.

    :param dense_layer: Keras Dense layer
    :param num_of_neurons: number of neurons that we want to delete from the layer
    :return: indices of neurons with the lowest std (len = num_of_neurons)
    """
    weights = dense_layer.get_weights()[0]   # without bias
    # w_abs = np.absolute(weights)
    indices_sorted = np.argsort(np.std(weights, axis=0))
    return indices_sorted[:num_of_neurons]


def parameter_pruning(model_name, X_train, y_train, X_test, y_test, pruning_share, num_of_epochs, batch_size):
    """
    Prune parameters/weights from pre-trained model.

    :param model_name: file name of the pre-trained model
    :param X_train: train input data
    :param y_train: train output labels
    :param X_test: test input data
    :param y_test: test output labels
    :param pruning_share: value between 0 and 1 that defines how many weights we want to prune
    :param num_of_epochs: number of epochs to train the model
    :param batch_size: batch size - number of samples per gradient update
    :return: Keras model after pruning
    """
    model = load_model(model_name)
    model.summary()

    # test accuracy on test data
    _, accuracy = model.evaluate(X_test, y_test, verbose=2)
    print('accuracy: ', round(accuracy * 100, 2))

    # delete some neurons from two Dense layers
    dense_1 = model.layers[1]
    indices_1 = best_channels2prune(dense_1, 1014)
    new_model = delete_channels(model, dense_1, indices_1)

    dense_2 = new_model.layers[3]
    indices_2 = best_channels2prune(dense_2, 1014)
    new_model = delete_channels(new_model, dense_2, indices_2)
    
    new_model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])
    new_model.summary()

    _, accuracy = new_model.evaluate(X_test, y_test, verbose=2)
    print('accuracy: ', round(accuracy * 100, 2))

    start = time.time()

    # re-train on train data with less weights
    new_model.fit(X_train, y_train, epochs=num_of_epochs, batch_size=batch_size)

    print(time.time() - start, 's')

    # test acc. on test data
    _, accuracy = new_model.evaluate(X_test, y_test, verbose=2)
    print('accuracy after re-training: ', round(accuracy * 100, 2))

    # new_model.save('saved_data/nn_model_compressed_20x.h5')


    '''
    # maybe delete only weights, not neurons
    # calculate threshold
    threshold = calculate_threshold(model, pruning_share)
    print(threshold)
    # create mask matrices
    # delete weights below threshold or set them to 0
    '''



if __name__ == '__main__':
    # to avoid cuDNN error (https://github.com/tensorflow/tensorflow/issues/24496)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    input_size = (28, 28)
    num_of_classes = 10

    num_of_epochs = 10
    batch_size = 600

    X_train, y_train, X_test, y_test = prepare_data(num_of_classes)
    # reshape to the right dimensions for CNN
    # X_train = X_train.reshape(X_train.shape[0], *input_size, 1)
    # X_test = X_test.reshape(X_test.shape[0], *input_size, 1)

    # model = cnn_model(input_size, num_of_classes)
    #
    # train_model(model, X_train, y_train, num_of_epochs, batch_size, validation_share=0.1)
    #
    # loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
    # print('accuracy: ', round(accuracy, 4))

    # model.save('saved_data/cnn_model.h5')

    pruning_share = 0.1
    # parameter_pruning('saved_data/cnn_model.h5', X_train, y_train, X_test, y_test, pruning_share, num_of_epochs, batch_size)
    parameter_pruning('saved_data/nn_model.h5', X_train, y_train, X_test, y_test, pruning_share, num_of_epochs, batch_size)



    # todo
    # size of each weight in bits --> 'numpy.float32'

    # Notes:
    # not a big difference between pruning random neurons or pruning neurons with the smallest std
    # usually in NN weight matrices have normal distributions with variance increasing towards output layer

