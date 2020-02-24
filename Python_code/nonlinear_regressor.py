
import numpy

from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
import numpy as np


def createNonlinearRegressionNeuralNet(A, b):
    '''
        Given a matrix A and set of labels b, create and return a
        nonlinear regression neural net
    '''
    dims = A.shape[1] # Number of columns

    # create model
    model = Sequential()

    # Input layer with dimension dims and hidden layer i with 128 neurons.
    model.add(Dense(128, input_dim=dims, activation='relu'))
    # Dropout of 20% of the neurons and activation layer.
    model.add(Dropout(.2))
    model.add(Activation("linear"))
    # Hidden layer j with 64 neurons plus activation layer.
    model.add(Dense(64, activation='relu'))
    model.add(Activation("linear"))
    # Hidden layer k with 64 neurons.
    model.add(Dense(64, activation='relu'))
    # Output Layer.
    model.add(Dense(1))

    # Model is derived and compiled using mean square error as loss
    # function, accuracy as metric and gradient descent optimizer.
    model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])

    # Training model with train data. Fixed random seed:
    numpy.random.seed(3)
    model.fit(A, b, nb_epoch=256, batch_size=2, verbose=2)

    return model


