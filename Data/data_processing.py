import os
import pandas as pd
import xarray as xr
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define paths
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"


def load_data(file_path):
    """Load dataset from different formats (CSV, JSON, NetCDF)."""
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".json"):
        return pd.read_json(file_path)
    elif file_path.endswith(".nc"):  # NetCDF file (used in climate science)
        dataset = xr.open_dataset(file_path)
        return dataset.to_dataframe().reset_index()
    else:
        print(f"Unsupported file format: {file_path}")
        return None


def clean_data(df):
    """Perform basic data cleaning (drop duplicates, handle missing values)."""
    df = df.drop_duplicates()

    # Fill missing numerical values with the median
    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical values with the most frequent value
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("Data cleaning complete.")
    return df


def transform_data(df):
    """Apply transformations like scaling and encoding."""
    # Standardize numerical columns
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Encode categorical variables
    encoder = LabelEncoder()
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = encoder.fit_transform(df[col])

    print("Data transformation complete.")
    return df


def process_data(filename):
    """Load, clean, and transform raw data, then save it."""
    file_path = os.path.join(RAW_DATA_PATH, filename)
    df = load_data(file_path)

    if df is not None:
        df_clean = clean_data(df)
        df_transformed = transform_data(df_clean)

        output_file = os.path.join(PROCESSED_DATA_PATH, "processed_" + filename)
        df_transformed.to_csv(output_file, index=False)
        print(f"Processed data saved: {output_file}")


if __name__ == "__main__":
    # Process all CSV, JSON, and NetCDF files in the raw data folder
    for file in os.listdir(RAW_DATA_PATH):
        if file.endswith((".csv", ".json", ".nc")):
            process_data(file)
