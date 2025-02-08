import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Load dataset
def load_data(file_path):
    """Load climate impact dataset."""
    df = pd.read_csv(file_path)
    return df


# Preprocess dataset
def preprocess_data(df):
    """Scale and reshape dataset for LSTM."""
    scaler = MinMaxScaler()
    features = df.drop(columns=['target'])  # Assuming 'target' is the impact variable
    target = df['target']

    features_scaled = scaler.fit_transform(features)

    # Reshape data for LSTM (samples, time steps, features)
    X = features_scaled.reshape(features_scaled.shape[0], 1, features_scaled.shape[1])
    y = target.values
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Build LSTM model
def build_iciaf_model(input_shape):
    """Create LSTM model for climate impact assessment."""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


if __name__ == "__main__":
    # Load and preprocess data
    FILE_PATH = "data/processed/climate_impact.csv"
    X_train, X_test, y_train, y_test = preprocess_data(load_data(FILE_PATH))

    # Build model
    model = build_iciaf_model(X_train.shape[1:])

    # Train model
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

    # Save model
    model.save("models/iciaf_model.h5")
    print("ICIAF Model saved successfully.")
