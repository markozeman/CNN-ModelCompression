from keras.datasets import cifar10, cifar100, mnist, fashion_mnist


def get_CIFAR_10():
    """
    Dataset of 50.000 32x32 color training images, labeled over 10 categories, and 10,000 test images.

    :return: tuple of X_train, y_train, X_test, y_test
    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    return X_train, y_train, X_test, y_test


def get_CIFAR_100():
    """
    Dataset of 50.000 32x32 color training images, labeled over 100 categories, and 10,000 test images.

    :return: tuple of X_train, y_train, X_test, y_test
    """
    (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')
    return X_train, y_train, X_test, y_test


def get_MNIST():
    """
    Dataset of 60.000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.

    :return: tuple of X_train, y_train, X_test, y_test
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return X_train, y_train, X_test, y_test


def get_fashion_MNIST():
    """
    Dataset of 60.000 28x28 grayscale images of 10 fashion categories, along with a test set of 10,000 images.

    :return: tuple of X_train, y_train, X_test, y_test
    """
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    return X_train, y_train, X_test, y_test


