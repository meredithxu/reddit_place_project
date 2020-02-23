
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

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

    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mae', 'mse'])

    EPOCHS = 1000

    history = model.fit(
        train_data, train_labels,
        epochs=EPOCHS, validation_split=0.2, verbose=0,
        callbacks=[tfdocs.modeling.EpochDots()])


    return model


