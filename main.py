from tensorflow import keras
from keras.api.models import load_model
from keras.api.utils import to_categorical
from train import load_test_data
from PIL import Image
import os
import numpy as np

model = load_model('models/cnn_model_2.keras')
print(model.summary())

infos = [x for x in os.walk('datasets/')][1:]
test_input = []
test_label = []

for dir, _, files in infos:
    for file in files:
        # file is a jpg file
        # open the file as image, convert to grayscale, resize to 28x28
        img = Image.open(f'{dir}/{file}').convert('L').resize((28, 28))
        # convert the image to a numpy array
        img = np.array(img)
        test_input.append(img)

        character = dir[-1] # uppercase ascii character
        label = ord(character) - ord('A') # convert to 0-25
        label = to_categorical(label, 26)
        test_label.append(label)

print(f'Test input: {len(test_input)}')
print(f'Test label: {len(test_label)}')

loss, acc = model.evaluate(np.array(test_input), np.array(test_label))
print(f'Loss: {loss}, Accuracy: {acc}')
