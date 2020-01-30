from xception import Xception_model
from datasets import *


if __name__ == '__main__':
    input_shape = (71, 71, 3)
    pooling = 'max'

    xception = Xception_model(input_shape, pooling)
    print(xception.summary())

    X_train, y_train, X_test, y_test = get_fashion_MNIST()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


