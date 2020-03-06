from keras import Sequential
from keras.engine.saving import load_model
from keras.layers import Dense, Activation, Lambda
from keras.callbacks import LambdaCallback, Callback
from plots import *
import tensorflow as tf
import numpy as np
from tensorflow_model_optimization.sparsity import keras as sparsity
from superposition import prepare_data


if __name__ == '__main__':
    input_size = (28, 28)
    num_of_classes = 10

    # num_of_epochs = 10
    batch_size = 600

    X_train, y_train, X_test, y_test = prepare_data(num_of_classes)
    # reshape to the right dimensions for CNN
    X_train = X_train.reshape(X_train.shape[0], *input_size, 1)
    X_test = X_test.reshape(X_test.shape[0], *input_size, 1)

    model = load_model('saved_data/cnn_model.h5')

    num_train_samples = X_train.shape[0]

    epochs = 4
    end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * epochs
    print(end_step)

    new_pruning_params = {
        'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                     final_sparsity=0.90,
                                                     begin_step=0,
                                                     end_step=end_step,
                                                     frequency=100)
    }

    new_pruned_model = sparsity.prune_low_magnitude(model, **new_pruning_params)
    new_pruned_model.summary()

    new_pruned_model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer='adam',
        metrics=['accuracy'])




