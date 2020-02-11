from keras import Sequential
from keras.layers import Dense, Activation, Lambda
from keras.callbacks import LambdaCallback, Callback
from plots import *
import tensorflow as tf
import numpy as np


def custom_activation(x):
    print('custom')
    return x


def my_activation(x, alpha):
    print('alpha')
    return x + alpha


class MyCallBack(Callback):

    def on_batch_begin(self, batch, logs=None):
        print('batch', batch)
        print('\nbegin', self.model.layers[1].get_weights())
        # print('\nparams', self.params)

        # change weights
        self.model.layers[1].set_weights([np.array([[1], [1.5], [1]]), np.array([1.0])])
        print('\nbegin_2', self.model.layers[1].get_weights())

    def on_batch_end(self, batch, logs=None):
        print('\nend', self.model.layers[1].get_weights())



if __name__ == '__main__':
    # Seed value
    seed_value = 0

    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.set_random_seed(seed_value)

    X = np.array([2, 4, 6, 8, 10])
    y = np.array([2, 3, 4, 5, 6])

    # callback_begin = LambdaCallback(on_epoch_begin=lambda epochs, logs: print('begin', model.layers[1].get_weights()) if epochs % 100 == 0 else print(''))
    # callback_end = LambdaCallback(on_epoch_end=lambda epochs, logs: print('end', model.layers[1].get_weights()) if epochs % 100 == 0 else print(''))

    call_back = MyCallBack()

    model = Sequential()
    model.add(Dense(3, input_shape=(1, ), activation=lambda x: my_activation(x, alpha=0.00)))
    model.add(Dense(1, activation=custom_activation))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.summary()

    history = model.fit(X, y, epochs=3001, verbose=2, batch_size=2, callbacks=[call_back])

    predictions = model.predict(np.array([-3, -1, 0, 1]))
    should_be = np.array([-0.5, 0.5, 1, 1.5])
    print('\nEvery number should be around 0: \n', np.squeeze(predictions) - should_be)

    # for layer in model.layers:
    #     print(layer)
    #     weights = layer.get_weights()
    #     print(weights)

'''
Epoch 3001/3001
begin [array([[-0.77509725],
       [-1.014928  ],
       [ 0.732941  ]], dtype=float32), array([0.27049163], dtype=float32)]
 - 0s - loss: 3.9836e-05 - accuracy: 1.0000
end [array([[-0.7750992],
       [-1.0149274],
       [ 0.7329403]], dtype=float32), array([0.2705016], dtype=float32)]
'''


