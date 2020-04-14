from superposition import *
from compression import train_model, best_channels2prune
from kerassurgeon.operations import delete_channels
from plots import plot_general


def deleting_neurons_1_task(X_train, y_train, input_size, num_of_units, num_of_classes, num_of_epochs, batch_size):
    """
    Normal training vs training with online compression of original MNIST task. Plot difference in validation accuracy.

    :param X_train: train input data
    :param y_train: train output labels
    :param input_size: tuple representing image size
    :param num_of_units: number of neurons in each hidden layer
    :param num_of_classes: number of different classes/output labels
    :param num_of_epochs: number of epochs to train the model
    :param batch_size: batch size - number of samples per gradient update
    :return: None
    """
    ### normal
    model = simple_model(input_size, num_of_units, num_of_classes)
    history = train_model(model, X_train, y_train, num_of_epochs, batch_size=batch_size, validation_share=0.1)
    val_acc_normal = np.array(history.history['val_accuracy']) * 100

    ### online compression
    val_acc_compression = []
    neurons_deleted = [524, 200, 200, 50]
    model = simple_model(input_size, num_of_units, num_of_classes)

    # few epochs without pruning
    epochs = math.floor(num_of_epochs / 5)
    history = train_model(model, X_train, y_train, epochs, batch_size=batch_size, validation_share=0.1)
    val_acc_compression.extend(np.array(history.history['val_accuracy']) * 100)

    for i, n in enumerate(neurons_deleted):
        # preserve only n neurons
        dense_1 = model.layers[1] if i == 0 else model.layers[2]    # keras.surgeon adds InputLayer after first pruning
        indices_1 = best_channels2prune(dense_1, n)
        model = delete_channels(model, dense_1, indices_1)

        dense_2 = model.layers[3]
        indices_2 = best_channels2prune(dense_2, n)
        model = delete_channels(model, dense_2, indices_2)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

        epochs = math.floor(num_of_epochs / 5)
        history = train_model(model, X_train, y_train, epochs, batch_size=batch_size, validation_share=0.1)
        val_acc_compression.extend(np.array(history.history['val_accuracy']) * 100)

    plot_general(val_acc_normal, val_acc_compression, ['normal val. acc.', 'online compression val. acc.'],
                 'Comparing validation accuracy in normal training and online compression training', 'epoch',
                 'validation accuracy (%)', [epochs * (i + 1) for i in range(len(neurons_deleted))],
                 min(val_acc_compression), max(val_acc_normal))


if __name__ == '__main__':
    input_size = (28, 28)
    num_of_units = 1024
    num_of_classes = 10

    num_of_tasks = 1       # todo - change to 50
    num_of_epochs = 20
    batch_size = 600

    if num_of_tasks == 1:
        X_train, y_train, X_test, y_test = prepare_data(num_of_classes)

        deleting_neurons_1_task(X_train, y_train, input_size, num_of_units, num_of_classes, num_of_epochs, batch_size)


