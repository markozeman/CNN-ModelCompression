import numpy as np
import tensorflow as tf
from keras.callbacks import Callback, LearningRateScheduler
from keras.engine.saving import load_model
from keras.models import Model
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from datasets import get_CIFAR_100
from math import floor, exp
from keras.utils.np_utils import to_categorical
from pre_trained_CNNs import ResNet50_model
from help_functions import zero_out_vector
from plots import plot_lr, plot_accuracies_over_time, plot_weights_histogram, show_image


class TestPerformanceCallback(Callback):
    """
    Callback class for testing normal model performance at the beginning of every epoch.
    """
    def __init__(self, X_test, y_test):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.accuracies = []

    def on_epoch_begin(self, epoch, logs=None):
        # evaluate on test images of the first task
        loss, accuracy = model.evaluate(self.X_test, self.y_test, verbose=2)
        self.accuracies.append(accuracy * 100)


class TestSuperpositionPerformanceCallback(Callback):
    """
    Callback class for testing superposition model performance at the beginning of every epoch.
    """
    def __init__(self, X_test, y_test, context_matrices, model, task_index):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.context_matrices = context_matrices
        self.model = model
        self.task_index = task_index
        self.accuracies = []

    def on_epoch_begin(self, epoch, logs=None):
        if self.task_index == 0:    # first task - we did not use context yet
            self.accuracies.append(-1)
            return

        # save current model weights (without bias node)
        curr_w_matrices = []
        curr_bias_vectors = []
        for layer in self.model.layers:
            curr_w_matrices.append(layer.get_weights()[0])
            curr_bias_vectors.append(layer.get_weights()[1])

        # temporarily change model weights to be suitable for first task (without bias node)
        for i, layer in enumerate(self.model.layers):
            # not multiplying with inverse because inverse is the same in binary superposition with {-1, 1} on the diagonal
            # using only element-wise multiplication on diagonal vectors for speed-up
            context_inverse_multiplied = np.diagonal(self.context_matrices[self.task_index][i])
            for task_i in range(self.task_index - 1, 0, -1):
                context_inverse_multiplied = np.multiply(context_inverse_multiplied, np.diagonal(self.context_matrices[task_i][i]))
            context_inverse_multiplied = np.diag(context_inverse_multiplied)

            layer.set_weights([curr_w_matrices[i] @ context_inverse_multiplied, curr_bias_vectors[i]])

        # evaluate on test images of the first task
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=2)
        self.accuracies.append(accuracy * 100)

        # change model weights back (without bias node)
        for i, layer in enumerate(self.model.layers):
            layer.set_weights([curr_w_matrices[i], curr_bias_vectors[i]])


lr_over_time = []   # global variable to store changing learning rates


def lr_scheduler(epoch, lr):
    """
    Learning rate scheduler function to set how learning rate changes each epoch.

    :param epoch: current epoch number
    :param lr: current learning rate
    :return: new learning rate
    """
    global lr_over_time
    lr_over_time.append(lr)
    decay_type = 'exponential'   # 'linear' or 'exponential'
    if decay_type == 'linear':
        lr -= 10 ** -5
    elif decay_type == 'exponential':
        initial_lr = 0.0001
        k = 0.07
        t = len(lr_over_time) % num_of_epochs     # to start each new task with the same learning rate as the first one
        lr = initial_lr * exp(-k * t)
    return max(lr, 0.000001)    # don't let learning rate go to 0


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
             [(X_train, y_train, X_test, y_test), (X_train, y_train, X_test, y_test), ...]
    """
    X_train, y_train, X_test, y_test = dataset_fun()
    train_sets = disjoint_datasets(X_train, y_train)
    test_sets = disjoint_datasets(X_test, y_test)
    return list(map(lambda x: (*x[0], *x[1]), zip(train_sets, test_sets)))


def simple_cnn(input_size, num_of_classes):
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
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def simple_nn(input_size, num_of_units, num_of_classes):
    """
    Create simple NN model with two hidden layers, each has 'num_of_units' neurons.

    :param input_size: vector input size
    :param num_of_units: number of neurons in each hidden layer
    :param num_of_classes: number of different classes/output labels
    :return: Keras model instance
    """
    model = Sequential()
    model.add(Dense(num_of_units, activation='relu', input_shape=(input_size, )))
    model.add(Dense(num_of_units, activation='relu'))
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
    :param batch_size: batch size - number of samples per gradient update
    :return: None
    """
    i = 0
    for X_train, y_train, X_test, y_test in datasets:
        X_train, y_train, X_test, y_test = prepare_data(X_train, y_train, X_test, y_test, input_size)
        model = simple_cnn(input_size, num_of_classes)
        model.fit(X_train, y_train, epochs=num_of_epochs, batch_size=batch_size, validation_split=0.1, verbose=2)
        # model.save('cifar-100_disjoint_dataset_models/cnn_model_%d.h5' % i)
        i += 1


def get_feature_vector_representation(datasets, input_size, proportion_0=0):
    """
    Load trained CNN models for 'datasets' and get representation vectors for all images
    after convolutional and pooling layers. Train and test labels do not change.

    :param datasets: list of disjoint datasets with corresponding train and test set
    :param input_size: image input shape
    :param proportion_0: share of zeros we want in vector to make it more sparse, default=0 which does not change original vector
    :return: 'datasets' images represented as feature vectors
             [(X_train_vectors, y_train, X_test_vectors, y_test), (X_train_vectors, y_train, X_test_vectors, y_test), ...]
    """
    i = 0
    vectors = []
    for X_train, y_train, X_test, y_test in datasets:
        X_train, y_train, X_test, y_test = prepare_data(X_train, y_train, X_test, y_test, input_size)

        model = load_model('cifar-100_disjoint_dataset_models/cnn_model_%d.h5' % i)
        i += 1

        # 6 is the index of Flatten layer which outputs feature vector after convolution and pooling
        intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[6].output)

        X_train_vectors = intermediate_layer_model.predict(X_train)
        X_test_vectors = intermediate_layer_model.predict(X_test)

        # # make input even more sparse, with 'proportion_0' of zero values
        for index in range(X_train_vectors.shape[0]):
            X_train_vectors[index] = zero_out_vector(X_train_vectors[index], proportion_0)
        for index in range(X_test_vectors.shape[0]):
            X_test_vectors[index] = zero_out_vector(X_test_vectors[index], proportion_0)
        # # plot_weights_histogram(X_train_vectors[0], 30)  # to test new weights distribution

        vectors.append((X_train_vectors, y_train, X_test_vectors, y_test))

    return vectors


def get_feature_vector_representation_ResNet50(datasets, input_size):
    """
    Use pre-trained ResNet50 CNN model for 'datasets' and get representation vectors for all images
    after convolutional and pooling layers. Train and test labels do not change.

    :param datasets: list of disjoint datasets with corresponding train and test set
    :param input_size: image input shape
    :return: 'datasets' images represented as feature vectors
             [(X_train_vectors, y_train, X_test_vectors, y_test), (X_train_vectors, y_train, X_test_vectors, y_test), ...]
    """
    vectors = []
    model = ResNet50_model(input_size, 'max')
    for X_train, y_train, X_test, y_test in datasets:
        X_train, y_train, X_test, y_test = prepare_data(X_train, y_train, X_test, y_test, input_size)

        X_train_vectors = model.predict(X_train)
        X_test_vectors = model.predict(X_test)

        vectors.append((X_train_vectors, y_train, X_test_vectors, y_test))

    return vectors


def train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, batch_size=32, validation_share=0.0,
                mode='normal', context_matrices=None, task_index=None):
    """
    Train and evaluate Keras model.

    :param model: Keras model instance
    :param X_train: train input data
    :param y_train: train output labels
    :param X_test: test input data
    :param y_test: test output labels
    :param num_of_epochs: number of epochs to train the model
    :param batch_size: batch size - number of samples per gradient update (default = 32)
    :param validation_share: share of examples to be used for validation (default = 0)
    :param mode: string for learning mode, important for callbacks - possible values: 'normal', 'superposition'
    :param context_matrices: multidimensional numpy array with random context (binary superposition), only used when mode = 'superposition'
    :param task_index: index of current task, only used when mode = 'superposition'
    :return: History object and 2 lists of test accuracies for every training epoch (normal and superposition)
    """
    test_callback = TestPerformanceCallback(X_test, y_test)
    test_superposition_callback = TestSuperpositionPerformanceCallback(X_test, y_test, context_matrices, model, task_index)
    lr_callback = LearningRateScheduler(lr_scheduler)

    callbacks = [lr_callback]
    if mode == 'normal':
        callbacks.append(test_callback)
    elif mode == 'superposition':
        callbacks.append(test_superposition_callback)

    history = model.fit(X_train, y_train, epochs=num_of_epochs, batch_size=batch_size, verbose=2,
                        validation_split=validation_share, callbacks=callbacks)
    return history, test_callback.accuracies, test_superposition_callback.accuracies


def context_multiplication(model, context_matrices, task_index):
    """
    Multiply current model weights with context matrices in each layer (without changing weights from bias node).

    :param model: Keras model instance
    :param context_matrices: multidimensional numpy array with random context (binary superposition)
    :param task_index: index of a task to know which context_matrices row to use
    :return: None (but model weights are changed)
    """
    for i, layer in enumerate(model.layers):
        curr_w = layer.get_weights()[0]
        curr_w_bias = layer.get_weights()[1]

        new_w = curr_w @ context_matrices[task_index][i]
        layer.set_weights([new_w, curr_w_bias])


def normal_training(model, datasets, num_of_epochs, num_of_tasks, batch_size=32):
    """
    Train model for 'num_of_tasks' tasks, each task is a different disjoint set of CIFAR-100 images.
    Check how accuracy for the first set of images is changing through tasks using normal training.

    :param model: Keras model instance
    :param datasets: list of disjoint datasets with corresponding train and test set
    :param num_of_epochs: number of epochs to train the model
    :param num_of_tasks: number of different tasks
    :param batch_size: batch size - number of samples per gradient update (default = 32)
    :return: list of test accuracies for 10 epochs for each task
    """
    original_accuracies = []

    # first training task - 10 classes of CIFAR-100 dataset
    X_train, y_train, X_test, y_test = datasets[0]     # these X_test and y_test are used for testing all tasks
    history, accuracies, _ = train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, batch_size, validation_share=0.1)
    original_accuracies.extend(accuracies)

    val_acc = np.array(history.history['val_accuracy']) * 100
    print('\nValidation accuracies: ', 'first', val_acc)

    # other training tasks
    for i in range(num_of_tasks - 1):
        print("\n\n i: %d \n" % i)

        X_train, y_train, _, _ = datasets[i + 1]    # use X_test and y_test from the first task to get its accuracy
        history, accuracies, _ = train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, batch_size, validation_share=0.1)
        original_accuracies.extend(accuracies)

        val_acc = np.array(history.history['val_accuracy']) * 100
        print('\nValidation accuracies: ', i, val_acc)

    return original_accuracies


def superposition_training(model, datasets, num_of_epochs, num_of_units, num_of_classes, num_of_tasks, batch_size=32):
    """
    Train model for 'num_of_tasks' tasks, each task is a different disjoint set of CIFAR-100 images.
    Check how accuracy for the first set of images is changing through tasks using superposition training.

    :param model: Keras model instance
    :param datasets: list of disjoint datasets with corresponding train and test set
    :param num_of_epochs: number of epochs to train the model
    :param num_of_units: number of neurons in each hidden layer
    :param num_of_classes: number of different classes/output labels
    :param num_of_tasks: number of different tasks
    :param batch_size: batch size - number of samples per gradient update (default = 32)
    :return: list of test accuracies for 10 epochs for each task (or validation accuracies for original task)
    """
    from superposition import get_context_matrices

    original_accuracies = []
    context_matrices = get_context_matrices(num_of_units, num_of_classes, num_of_tasks)

    # first training task - 10 classes of CIFAR-100 dataset
    X_train, y_train, X_test, y_test = datasets[0]  # these X_test and y_test are used for testing all tasks
    history, _, _ = train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, batch_size, validation_share=0.1,
                                mode='superposition', context_matrices=context_matrices, task_index=0)

    val_acc = np.array(history.history['val_accuracy']) * 100
    print('\nValidation accuracies: ', 'first', val_acc)

    original_accuracies.extend(val_acc)

    # other training tasks
    for i in range(num_of_tasks - 1):
        print("\n\n i: %d \n" % i)

        # multiply current weights with context matrices for each layer (without changing weights from bias node)
        context_multiplication(model, context_matrices, i + 1)

        X_train, y_train, _, _ = datasets[i + 1]    # use X_test and y_test from the first task to get its accuracy
        history, _, accuracies = train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, batch_size, validation_share=0.1,
                                             mode='superposition', context_matrices=context_matrices, task_index=i + 1)
        original_accuracies.extend(accuracies)

        val_acc = np.array(history.history['val_accuracy']) * 100
        print('\nValidation accuracies: ', i, val_acc)

    return original_accuracies


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

    # num_of_epochs = 50
    # batch_size = 50
    # train_CNNs(disjoint_sets, input_size, num_of_classes, num_of_epochs, batch_size)

    datasets_vectors = get_feature_vector_representation(disjoint_sets, input_size, proportion_0=0)
    # datasets_vectors = get_feature_vector_representation_ResNet50(disjoint_sets, input_size)

    input_size = datasets_vectors[0][0].shape[1]
    num_of_units = 1000

    num_of_tasks = len(datasets_vectors)
    num_of_epochs = 10
    batch_size = 500

    train_normal = True
    train_superposition = True

    if train_normal:
        model = simple_nn(input_size, num_of_units, num_of_classes)

        acc_normal = normal_training(model, datasets_vectors, num_of_epochs, num_of_tasks, batch_size=batch_size)

        if not train_superposition:
            plot_lr(lr_over_time)
            plot_accuracies_over_time(acc_normal, np.zeros(len(acc_normal)))

    if train_superposition:
        lr_over_time = []  # re-initiate learning rate

        model = simple_nn(input_size, num_of_units, num_of_classes)

        acc_superposition = superposition_training(model, datasets_vectors, num_of_epochs, num_of_units, num_of_classes, num_of_tasks, batch_size=batch_size)

        if not train_normal:
            plot_lr(lr_over_time)
            plot_accuracies_over_time(np.zeros(len(acc_superposition)), acc_superposition)
        else:
            plot_accuracies_over_time(acc_normal, acc_superposition)



    ### Notes:
    # num_of_epochs = 75,  batch_size = 50
    # average end validation accuracy: 0.65 --> simple_cnn function
    # average end validation accuracy: 0.70 --> example model at: https://keras.io/examples/cifar10_cnn/

    # vector of length 1600 representing each image (all values are non-negative)
    # around 25% of feature vectors values are 0

