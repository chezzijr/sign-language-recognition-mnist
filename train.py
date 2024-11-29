import pandas as pd
import numpy as np
import config
import keras
from os import path
from keras.api.callbacks import EarlyStopping
from keras.src.legacy.preprocessing.image import ImageDataGenerator

TRAIN_DATA_PATH = path.join(config.DATA_PATH, "sign_mnist_train.csv")
TEST_DATA_PATH = path.join(config.DATA_PATH, "sign_mnist_test.csv")

def _load_all_data():
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    test_data = pd.read_csv(TEST_DATA_PATH)

    label_train = train_data["label"]
    X_train = train_data.drop("label", axis=1)
    X_train = np.array(X_train).reshape(-1, 28, 28, 1)
    y_train = keras.utils.to_categorical(label_train, config.NUM_CLASSES)

    image_gen = ImageDataGenerator(
        rescale=1.0 / 255,  # easier for network to interpret numbers in range [0,1]
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=10,
        validation_split=config.VALIDATION_RATIO,
    )

    train_generator = image_gen.flow(
        X_train,
        y_train,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        subset="training",
        seed=42,
    )

    valid_generator = image_gen.flow(
        X_train,
        y_train,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        subset="validation",
    )

    label_test = test_data["label"]
    X_test = test_data.drop("label", axis=1)

    X_test = np.array(X_test).reshape(-1, 28, 28, 1)
    X_test = X_test / 255.0
    y_test = keras.utils.to_categorical(label_test, config.NUM_CLASSES)
    return train_generator, valid_generator, X_train, y_train, X_test, y_test

def data_generator():
    train_generator, valid_generator, *_ = _load_all_data()
    return train_generator, valid_generator

def load_train_data():
    _, _, X_train, y_train, _, _ = _load_all_data()
    return X_train, y_train

def load_test_data():
    _, _, _, _, X_test, y_test = _load_all_data()
    return X_test, y_test


def train(model, train_generator, valid_generator):
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=config.EPOCHS,
        callbacks=[EarlyStopping(patience=5, monitor="loss")],
    )
    return model

def evaluate(model, X_test, y_test):
    loss, acc = model.evaluate(X_test, y_test)
    return loss, acc

def save_model(model, name):
    model.save(path.join(config.MODEL_PATH, name))
