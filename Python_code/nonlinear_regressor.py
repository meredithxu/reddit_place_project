import tensorflow as tf
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
import numpy as np
import random


def createNonlinearRegressionNeuralNet(train_A, train_b, val_A, val_b):
    '''
        Given a matrix A and set of labels b, create and return a
        nonlinear regression neural net
    '''
    SEED = 1021
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    dims = A.shape[1] # Number of columns

    # create model
    model = Sequential()

    # Input layer with dimension dims and hidden layer i with 128 neurons.
    model.add(Dense(128, input_dim=dims, activation='relu'))
    # Dropout of 20% of the neurons and activation layer.
    model.add(Dropout(.2))
    # Hidden layer j with 64 neurons plus activation layer.
    model.add(Dense(64, activation='relu'))
    # Hidden layer k with 64 neurons.
    model.add(Dense(64, activation='relu'))
    # Output Layer.
    model.add(Dense(1))

    # Model is derived and compiled using mean square error as loss
    # function, accuracy as metric and gradient descent optimizer.
    model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])

    # Training model with train data.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            monitor='val_loss',
            # "no longer improving" being defined as "no better than 1e-2 less"
            min_delta=1e-2,
            # "no longer improving" being further defined as "for at least 2 epochs"
            patience=2,
            verbose=1)
    ]
    model.fit(train_A, train_b, epochs=256, callbacks=callbacks, batch_size=2, validation_data=(val_A, val_b), verbose=2)

    return model


