
def Xception_model(input_shape, pooling):
    """
    Get Xception model pre-trained on ImageNet data.

    :param input_shape: (height, width, channels), height and width should be at least 71
    :param pooling: pooling mode for feature extraction (None, 'avg' or 'max')
    :return: Keras Xception model
    """
    from keras.applications.xception import Xception
    return Xception(include_top=False, weights='imagenet', input_shape=input_shape, pooling=pooling)


def ResNet50_model(input_shape, pooling):
    """
    Get ResNet50 model pre-trained on ImageNet data.

    :param input_shape: (height, width, channels)
    :param pooling: pooling mode for feature extraction (None, 'avg' or 'max')
    :return: Keras ResNet50 model
    """
    from keras.applications.resnet import ResNet50
    return ResNet50(include_top=False, weights='imagenet', input_shape=input_shape, pooling=pooling)


