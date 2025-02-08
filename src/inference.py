import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Define paths
MODEL_PATH = "../models/climate_model.h5"
SAMPLE_DATA_PATH = "../data/processed/climate_data_processed.csv"


def load_model():
    """Load the trained deep learning model."""
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_input(sample_data):
    """Preprocess input data for prediction."""
    scaler = StandardScaler()
    sample_data = scaler.fit_transform(sample_data)
    return sample_data.reshape(sample_data.shape[0], 1, sample_data.shape[1])


def predict_climate_impact(sample_data):
    """Run predictions using the trained model."""
    model = load_model()

    processed_input = preprocess_input(sample_data)
    predictions = model.predict(processed_input)

    return predictions


if __name__ == "__main__":
    # Load a sample dataset (for demonstration)
    df = pd.read_csv(SAMPLE_DATA_PATH)
    sample_input = df.iloc[:5, :-1].values  # Selecting first 5 samples for inference

    predictions = predict_climate_impact(sample_input)
    print("Predictions:", predictions)
