from superposition import simple_model, prepare_data


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
    pass




if __name__ == '__main__':
    input_size = (28, 28)
    num_of_units = 1024
    num_of_classes = 10

    num_of_epochs = 10
    batch_size = 600

    X_train, y_train, X_test, y_test = prepare_data(num_of_classes)

    model = simple_model(input_size, num_of_units, num_of_classes)

    train_model(model, X_train, y_train, num_of_epochs, batch_size, validation_share=0.1)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
    print('loss: ', loss)
    print('accuracy: ', accuracy)





