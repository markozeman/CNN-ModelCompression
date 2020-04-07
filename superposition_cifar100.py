import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from datasets import get_CIFAR_100
from math import floor
from keras.utils.np_utils import to_categorical


def disjoint_datasets(X, y):
    """
    Separate bigger dataset to 10 smaller datasets.

    :param X: model input data
    :param y: model output data / label
    :return: 10 disjoint datasets
    """
    sets = [([], []) for _ in range(10)]
    for image, label in zip(X, y):
        index = int(floor(label[0] / 10))
        sets[index][0].append(image)
        sets[index][1].append(to_categorical(label[0] % 10, 10))
    return sets


def make_disjoint_datasets(dataset_fun=get_CIFAR_100):
    """
    Make 10 disjoint datasets of the same size from CIFAR-100 or other 'dataset_fun' dataset.

    :param dataset_fun: function that returns specific dataset (default is CIFAR-100 dataset)
    :return: list of 10 disjoint datasets with corresponding train and test set
    """
    X_train, y_train, X_test, y_test = dataset_fun()
    train_sets = disjoint_datasets(X_train, y_train)
    test_sets = disjoint_datasets(X_test, y_test)
    return list(map(lambda x: (*x[0], *x[1]), zip(train_sets, test_sets)))


def simple_model(input_size, num_of_classes):
    """
    Create simple CNN model.

    :param input_size: image input shape
    :param num_of_classes: number of different classes/output labels
    :return: Keras model instance
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_size))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


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
    X_train = np.array(X_train).astype(dtype=np.float64)
    X_test = np.array(X_test).astype(dtype=np.float64)
    X_train /= 255
    X_test /= 255

    # reshape to the right dimensions for CNN
    X_train = X_train.reshape(X_train.shape[0], *input_size)
    X_test = X_test.reshape(X_test.shape[0], *input_size)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test


def train_CNNs(datasets, input_size, num_of_classes, num_of_epochs, batch_size):
    """
    Train different CNN for each dataset in 'datasets' and save models.

    :param datasets: list of disjoint datasets with corresponding train and test set
    :param input_size: image input shape
    :param num_of_classes: number of different classes/output labels
    :param num_of_epochs: number of epochs to train each model
    :param batch_size:  batch size - number of samples per gradient update
    :return: None
    """
    for X_train, y_train, X_test, y_test in datasets:
        X_train, y_train, X_test, y_test = prepare_data(X_train, y_train, X_test, y_test, input_size)

        model = simple_model(input_size, num_of_classes)
        model.fit(X_train, y_train, epochs=num_of_epochs, batch_size=batch_size, validation_split=0.1, verbose=2)

        # save model

        break



if __name__ == '__main__':
    # to avoid cuDNN error (https://github.com/tensorflow/tensorflow/issues/24496)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    disjoint_sets = make_disjoint_datasets()

    # for s in disjoint_sets:
    #     X_train, y_train, X_test, y_test = s
    #     print(X_train[0].shape, y_train[0].shape, X_test[0].shape, y_test[0].shape)
    #     print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    #     print()

    input_size = (32, 32, 3)
    num_of_classes = 10

    num_of_epochs = 50
    batch_size = 50

    train_CNNs(disjoint_sets, input_size, num_of_classes, num_of_epochs, batch_size)




    # todo
    # from CIFAR-100 dataset make 10 smaller disjoint datasets, so we get 10 tasks of CIFAR-10
    # try more complex CNN
    # for each of CIFAR-10 datasets build and train separate CNN
    # for each of 10 CNN models extract weights from convolutional layers only
    # put all images through corresponding convolutional layers to get vector representation
    # this vector representation is an input to the multilayer perceptron
    # consecutively train each of 10 tasks in the network with 1.000 neurons in both hidden layers to the desirable validation accuracy
    # compare how test accuracy on the first CIFAR-10 task is decreasing with or without using superposition




