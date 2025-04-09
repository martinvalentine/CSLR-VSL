import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Model Definition
model = Sequential([
    LSTM(64, return_sequences=True, activation='tanh', input_shape=(30, 1662)),
    LSTM(128, return_sequences=True, activation='tanh'),
    LSTM(64, return_sequences=False, activation='tanh'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

# Load the models
def load_model(model, model_name):
    model.load_weights(model_name)
    return model