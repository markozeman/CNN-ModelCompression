"""
Based on article 'Superposition of many models into one':
https://arxiv.org/pdf/1902.05522.pdf
"""
from datasets import *
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils.np_utils import to_categorical

from plots import *


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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, batch_size=None):
    """
    Train Keras model.

    :param model: Keeras model instance
    :param X_train: train input data
    :param y_train: train output labels
    :param X_test: test input data
    :param y_test: test output labels
    :param num_of_epochs: number of epochs to train the model
    :param batch_size: batch size - number of samples per gradient update (default = 32)
    :return: History object
    """
    history = model.fit(X_train, y_train, epochs=num_of_epochs, batch_size=batch_size, verbose=2, validation_split=0.2)
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)
    print('\nEvaluation accuracy: %.2f' % (accuracy * 100))
    return history


if __name__ == '__main__':
    input_size = (28, 28)
    num_of_units = 100
    num_of_classes = 10
    model = simple_model(input_size, num_of_units, num_of_classes)

    X_train, y_train, X_test, y_test = get_MNIST()
    y_train = to_categorical(y_train, num_classes=num_of_classes)   # one-hot encode
    y_test = to_categorical(y_test, num_classes=num_of_classes)     # one-hot encode

    num_of_epochs = 25
    history = train_model(model, X_train, y_train, X_test, y_test, num_of_epochs)
    plot_fit_history(history)


