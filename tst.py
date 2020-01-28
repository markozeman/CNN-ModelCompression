from xception import Xception_model


if __name__ == '__main__':
    input_shape = (71, 71, 3)
    pooling = 'max'

    xception = Xception_model(input_shape, pooling)
    print(xception.summary())

