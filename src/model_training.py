import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Load processed dataset
PROCESSED_DATA_PATH = "../data/processed/climate_data_processed.csv"
MODEL_SAVE_PATH = "../models/climate_model.h5"


def load_processed_data():
    """Load preprocessed climate dataset."""
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Assuming the last column is the target variable
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_model(input_shape):
    """Define and compile an LSTM-based deep learning model."""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(1, input_shape)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_model():
    """Train the deep learning model."""
    X_train, X_test, y_train, y_test = load_processed_data()

    # Reshape for LSTM (samples, time steps, features)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    model = build_model(X_train.shape[2])

    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

    # Save model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_model()
