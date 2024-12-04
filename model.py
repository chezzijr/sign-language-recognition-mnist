from tensorflow import keras 
from keras import layers
from keras.api.regularizers import L2
from keras.api.callbacks import EarlyStopping
from keras.api import Sequential
import config


def cnn_model_1():
    model = Sequential([
        layers.Conv2D(6, (5, 5), padding="same", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.ReLU(),
        layers.Conv2D(16, (5, 5), padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.ReLU(),
        layers.Flatten(),
        layers.Dense(128, activation="relu", kernel_regularizer=L2()),
        layers.Dense(64, activation="relu", kernel_regularizer=L2()),
        layers.Dropout(0.2),
        layers.Dense(26, activation="softmax"),
    ])
    return model


def cnn_model_2():
    model = Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (5, 5), strides=2, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(config.DROPOUT_RATE),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (5, 5), strides=2, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(config.DROPOUT_RATE),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(config.DROPOUT_RATE),
        layers.Dense(config.NUM_CLASSES, activation="softmax"),
    ])
    return model
