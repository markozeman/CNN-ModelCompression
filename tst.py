from xception import Xception_model
from datasets import *
from plots import *


if __name__ == '__main__':
    # input_shape = (71, 71, 3)
    # pooling = 'max'
    #
    # xception = Xception_model(input_shape, pooling)
    # print(xception.summary())

    X_train, y_train, X_test, y_test = get_CIFAR_100()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    ind = 10
    print(X_train[ind].shape, y_train[ind])

    show_image(X_train[ind])



