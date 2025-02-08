import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Define file paths
RAW_DATA_PATH = "../data/raw/climate_data.csv"
PROCESSED_DATA_PATH = "../data/processed/climate_data_processed.csv"


def load_data(file_path):
    """Load raw climate dataset."""
    return pd.read_csv(file_path)


def clean_data(df):
    """Clean data by handling missing values and duplicates."""
    df = df.drop_duplicates()

    # Fill missing numerical values with median
    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical values with mode
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("Data cleaning complete.")
    return df


def transform_data(df):
    """Scale numerical features and encode categorical variables."""
    scaler = StandardScaler()
    encoder = LabelEncoder()

    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    df[num_cols] = scaler.fit_transform(df[num_cols])
    for col in cat_cols:
        df[col] = encoder.fit_transform(df[col])

    print("Data transformation complete.")
    return df


def process_data():
    """Load, clean, transform, and save processed data."""
    df = load_data(RAW_DATA_PATH)
    df_clean = clean_data(df)
    df_transformed = transform_data(df_clean)
    df_transformed.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    process_data()
