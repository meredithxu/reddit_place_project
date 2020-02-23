
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from keras.layers import Dense
from keras.models import Sequential
import numpy as np


# To install tensorflow_docs: pip install git+https://github.com/tensorflow/docs
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling


def createNonlinearRegressionNeuralNet(A, b):
    '''
        Given a matrix A and set of labels b, create and return a
        nonlinear regression neural net
    '''

    dataset = np.array(A), np.array(b)

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_data = train_dataset[0]
    train_labels = train_dataset[1]

    test_data = test_dataset[0]
    test_labels = test_dataset[1]

    # create model
    model = Sequential()
    model.add(Dense(20, activation="tanh", input_dim=5,
                    kernel_initializer="uniform"))
    model.add(Dense(1, activation="linear", kernel_initializer="uniform"))

    # Compile model
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(train_data, train_labels, epochs=1000, batch_size=10,  verbose=2)

    return model


