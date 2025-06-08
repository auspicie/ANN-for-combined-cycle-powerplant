# src/preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def split_data(df, target_column='PE', test_size=0.2, random_state=42):
    """
    Splits the DataFrame into features (X) and target (y),
    then further splits them into training and testing sets.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for random splitting for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Data split: X_train shape {X_train.shape}, X_test shape {X_test.shape}")
    return X_train, X_test, y_train, y_test

def create_preprocessing_pipeline():
    """
    Creates a scikit-learn pipeline for data preprocessing.
    Currently includes only StandardScaler.

    Returns:
        sklearn.pipeline.Pipeline: A preprocessing pipeline.
    """
    # For this dataset, StandardScaler is generally sufficient.
    # You could add other steps like dimensionality reduction here if needed.
    pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    return pipeline

if __name__ == '__main__':
    # Example usage (requires dummy data or actual data loaded)
    # This block won't run correctly standalone without 'data/Folds5x2_pp.xlsx'
    # and load_ccpp_data from data_loader.py
    try:
        from data_loader import load_ccpp_data
        dummy_data = load_ccpp_data()
        if dummy_data is not None:
            X_train, X_test, y_train, y_test = split_data(dummy_data)
            preprocessor = create_preprocessing_pipeline()
            
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            print(f"\nProcessed X_train shape: {X_train_processed.shape}")
            print(f"Processed X_test shape: {X_test_processed.shape}")
            print("\nFirst 5 rows of scaled X_train:")
            print(pd.DataFrame(X_train_processed, columns=X_train.columns).head())
        else:
            print("Cannot run preprocessing example without loading data.")
    except ImportError:
        print("Run this from the main.py or ensure data_loader is accessible.")
    except Exception as e:
        print(f"An error occurred in preprocessing example: {e}")