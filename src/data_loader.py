import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import sys
import os

# Add src to system path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

def load_data():
    """
    Loads Train and Test data from CSV/TXT files.
    """
    print(f"Loading data from {config.TRAIN_PATH}...")
    train_df = pd.read_csv(config.TRAIN_PATH, names=config.NSL_KDD_COLUMNS)
    test_df = pd.read_csv(config.TEST_PATH, names=config.NSL_KDD_COLUMNS)
    
    return train_df, test_df

def preprocess_data(train_df, test_df):
    """
    Applies MinMax Scaling to numeric features and One-Hot Encoding to categorical.
    Encodes labels to 0 (Benign) and 1 (Malicious).
    """
    print("Preprocessing data...")
    
    # 1. Separate Features and Targets
    # Drop 'difficulty_level' (it's not a feature)
    X_train = train_df.drop(columns=["label", "difficulty_level"])
    y_train = train_df["label"]
    
    X_test = test_df.drop(columns=["label", "difficulty_level"])
    y_test = test_df["label"]

    # 2. Binary Classification: Map all attacks to 1, normal to 0
    # 'normal' is the label for benign traffic in NSL-KDD
    y_train_bin = y_train.apply(lambda x: 0 if x == 'normal' else 1)
    y_test_bin = y_test.apply(lambda x: 0 if x == 'normal' else 1)
    
    # 3. Define Preprocessing Pipeline
    # We use ColumnTransformer to apply different preprocessing to different columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), config.NUMERIC_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), config.CAT_COLS)
        ]
    )
    
    # 4. Fit on TRAIN, Transform on TEST
    # IMPORTANT: Never fit on test data (data leakage)
    print("Fitting preprocessor on training data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 5. Save the preprocessor for future use (e.g., during inference)
    joblib.dump(preprocessor, os.path.join(config.PROCESSED_DATA_DIR, 'preprocessor.pkl'))
    
    print(f"Data Processed. Train Shape: {X_train_processed.shape}, Test Shape: {X_test_processed.shape}")
    
    return X_train_processed, y_train_bin.values, X_test_processed, y_test_bin.values

if __name__ == "__main__":
    train_df, test_df = load_data()
    X_train, y_train, X_test, y_test = preprocess_data(train_df, test_df)
    
    # Save processed numpy arrays for fast loading later
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'y_test.npy'), y_test)
    
    print("Preprocessing Complete. Files saved to data/processed/")