import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Model Definition

# Model Definition
model = Sequential([
    LSTM(64, return_sequences=True, activation='tanh', input_shape=(40, 1662)),
    LSTM(128, return_sequences=True, activation='tanh'),
    Dropout(0.3),
    LSTM(64, return_sequences=False, activation='tanh'),
    Dropout(0.3),  # Increased for better regularization
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(5, activation='softmax')
])


# Load the models
def load_model(model, model_name):
    model.load_weights(model_name)
    return model