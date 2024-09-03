import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GaussianNoise
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Define the Denoising Autoencoder using Keras
def build_denoising_autoencoder(input_dim, dropout_rate=0.5):
    input_layer = Input(shape=(input_dim,), name='input')

    # Encoder
    encoded = Dense(256, activation='relu')(input_layer)
    encoded = Dropout(dropout_rate)(encoded)
    encoded = Dense(128, activation='relu')(encoded)
    encoded = Dropout(dropout_rate)(encoded)

    # Decoder
    decoded = Dense(256, activation='relu')(encoded)
    decoded = Dropout(dropout_rate)(decoded)
    decoded = Dense(input_dim, activation='tanh', name='output')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    return autoencoder

def build_dynamic_autoencoder(input_dim, hidden_layers, hidden_size_factor=0.2, noise=None):
    input_ = Input(shape=(input_dim,), name='input')
    x = input_
    layer_size = 256

    if noise is not None:
        x = GaussianNoise(noise)(x)

    for i in range(hidden_layers):
        if isinstance(hidden_size_factor, list):
            factor = hidden_size_factor[i]
        else:
            factor = hidden_size_factor
        x = Dense(int(input_dim * factor), activation='relu', name=f'hid{i + 1}')(x)
        x = Dropout(0.5)(x)

    output = Dense(input_dim, activation='tanh', name='output')(x)

    model = Model(inputs=input_, outputs=output)

    return model

def add_noise(data, noise_factor=0.5):
    noisy_data = data + noise_factor * np.random.randn(*data.shape)
    noisy_data = np.clip(noisy_data, 0., 1.)  # Ensure the values are within [0, 1] range if normalized
    return noisy_data

# Train the Denoising Autoencoder
def train_denoising_autoencoder(model, train_data, val_data, epochs=50, batch_size=64, learning_rate=0.0001, noise_factor=0.5, patience=10):
    optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.99, epsilon=1e-8)
    model.compile(optimizer=optimizer, loss='mse')

    # Add noise to training and validation data
    #noisy_train_data = add_noise(train_data, noise_factor)
    #noisy_val_data = add_noise(val_data, noise_factor)

    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)

    history = model.fit(
        train_data, train_data,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(val_data, val_data),
        callbacks=[early_stopping]
    )
    return history