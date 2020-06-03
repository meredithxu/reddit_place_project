import tensorflow as tf
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras import optimizers
import numpy as np
import random


def createNonlinearRegressionNeuralNet(A, b, 
                                        train_proportion = 0.9, 
                                        first_nodes = 128, 
                                        second_nodes = 64, 
                                        dropout = 0.2, 
                                        learning_rate = 0.001, 
                                        batch_size = 2, 
                                        epochs = 256):
    '''
        Given a matrix A and set of labels b, create and return a
        nonlinear regression neural net
    '''
    SEED = 1021
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Split data into training and validation
    rand = np.random.rand(A.shape[0], A.shape[1])
    np.random.shuffle(rand)

    num_training = int(A.shape[0] * train_proportion)

    train_A = A[:num_training]
    train_b = b[:num_training]
    val_A = A[num_training:]
    val_b = b[num_training:]

    dims = train_A.shape[1] # Number of columns

    # create model
    model = Sequential()

    # Hidden layer j with 128 neurons plus activation layer.
    model.add(Dense(first_nodes, input_dim=dims, activation='relu'))
    model.add(Dropout(dropout))
    # Hidden layer k with 64 neurons.
    model.add(Dense(second_nodes, activation='relu'))
    model.add(Dropout(dropout))
    # Output Layer.
    model.add(Dense(1))

    adam = optimizers.Adam(clipnorm=1, learning_rate=learning_rate)
    # Model is derived and compiled using mean square error as loss
    # function, accuracy as metric and gradient descent optimizer.
    model.compile(loss='mse', optimizer=adam)

    # Training model with train data.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            monitor='val_loss',
            # "no longer improving" being defined as "no better than 1e-2 less"
            min_delta=1e-2,
            # "no longer improving" being further defined as "for at least 2 epochs"
            patience=5,
            verbose=1)
    ]

    model.fit(train_A, train_b, epochs=epochs, callbacks=callbacks, batch_size=batch_size, validation_data=(val_A, val_b), verbose=2)

    return model


