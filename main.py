from keras.api.models import load_model
from train import load_test_data

model = load_model('models/cnn_model_2.keras')
print(model.summary())

X, y = load_test_data()
loss, accuracy = model.evaluate(X, y)
print('Loss:', loss)
print('Accuracy:', accuracy)
