# src/data_loader.py
import pandas as pd

def load_ccpp_data(filepath='data/Folds5x2_pp.xlsx', sheet_name=0):
    """
    Loads the Combined Cycle Power Plant dataset from an Excel file.

    Args:
        filepath (str): Path to the Excel file.
        sheet_name (int or str): Sheet name or index to load.
                                  The dataset is often provided across 5 sheets for 5x2 fold CV.
                                  We use the first sheet (index 0) by default for simplicity.

    Returns:
        pandas.DataFrame: Loaded dataset, or None if an error occurs.
    """
    try:
        # If your Excel file has a specific sheet name, replace `0` with that name (e.g., 'Sheet1')
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        # Clean column names by stripping whitespace
        df.columns = df.columns.str.strip()
        print(f"Data loaded successfully from {filepath} (Sheet: {sheet_name}). Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return None

if __name__ == '__main__':
    # Example usage:
    data = load_ccpp_data()
    if data is not None:
        print("\nFirst 5 rows of data:")
        print(data.head())
        print("\nData Info:")
        data.info()