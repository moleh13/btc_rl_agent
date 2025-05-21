# /Users/melihkarakose/Desktop/EC 581/btc_rl/rl_trader/src/data_manager.py

import pandas as pd
import numpy as np
import os

# These are the 33 features the agent will see, including OHLCV.
# Make sure these column names exactly match your CSV file.
EXPECTED_FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume',
    'log_returns', 'volatility_30_period', 'RSI14', 'SMA50',
    'EMA12', 'EMA26', 'EMA100', 'EMA200',
    'MACD_line', 'MACD_signal_line', 'MACD_histogram',
    '%K', '%D', 'ATR',
    'BB_Middle_Band', 'BB_Upper_Band', 'BB_Lower_Band', 'BB_wWidth',
    'price_vs_bb_upper', 'price_vs_bb_lower',
    'OBV', 'VWAP', 'ROC10',
    'Hour_of_Day', 'Day_of_Week', 'Day_of_Month', 'Month_of_Year', 'Week_of_Year'
]

def load_and_preprocess_data(csv_file_path,
                             date_column_name='timestamp',  # Adjust if your date column has a different name
                             expected_features=None):
    """
    Loads, preprocesses, and splits the BTC hourly data.

    Args:
        csv_file_path (str): Path to the CSV file.
        date_column_name (str): The name of the column containing datetime information.
        expected_features (list): List of column names that are expected to be features.

    Returns:
        tuple: (train_df, test_df, feature_columns)
               train_df: DataFrame for training.
               test_df: DataFrame for testing.
               feature_columns: List of actual feature column names used.
    """
    if expected_features is None:
        expected_features = EXPECTED_FEATURE_COLUMNS

    print(f"Loading data from: {csv_file_path}")
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file not found at {csv_file_path}")

    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

    print("Data loaded successfully.")

    # --- 1. Date Handling ---
    if date_column_name not in df.columns:
        raise ValueError(
            f"Date column '{date_column_name}' not found in CSV. "
            f"Available columns: {df.columns.tolist()}"
        )
    try:
        df[date_column_name] = pd.to_datetime(df[date_column_name])
    except Exception as e:
        raise ValueError(f"Error converting date column '{date_column_name}' to datetime: {e}")

    df.set_index(date_column_name, inplace=True)
    df.sort_index(inplace=True) # Ensure chronological order
    print(f"Data indexed by '{date_column_name}'. Date range: {df.index.min()} to {df.index.max()}")

    # --- 2. Feature Verification and Selection ---
    # Clean column names (e.g., remove leading/trailing spaces)
    df.columns = df.columns.str.strip()

    missing_expected_cols = [col for col in expected_features if col not in df.columns]
    if missing_expected_cols:
        raise ValueError(
            f"The following expected feature columns are missing from the CSV: {missing_expected_cols}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Select only the expected features
    df_features = df[expected_features].copy() # Use .copy() to avoid SettingWithCopyWarning
    print(f"Selected {len(expected_features)} feature columns.")

    # --- 3. Data Type Conversion and Cleaning ---
    for col in expected_features:
        # Attempt to convert to numeric, coercing errors will turn non-numeric to NaN
        df_features[col] = pd.to_numeric(df_features[col], errors='coerce')

    # --- 4. Handle Missing Values (NaNs) ---
    nan_counts_before = df_features.isnull().sum()
    nan_cols_before = nan_counts_before[nan_counts_before > 0]
    if not nan_cols_before.empty:
        print(f"NaN counts per column before filling:\n{nan_cols_before}")
    else:
        print("No NaNs found before filling.")

    # Fill NaNs: first forward, then backward
    # This is important for indicators that have NaNs at the beginning (e.g., long SMAs)
    df_features.ffill(inplace=True)
    df_features.bfill(inplace=True)

    nan_counts_after = df_features.isnull().sum()
    nan_cols_after = nan_counts_after[nan_counts_after > 0]
    if not nan_cols_after.empty:
        print(f"Warning: NaNs still present after ffill/bfill in columns:\n{nan_cols_after}")
        print("This might happen if entire columns are NaN or NaNs exist at the very start AND end of the dataset.")
        print("Consider dropping these rows/columns or using more sophisticated imputation if this is an issue.")
        # Depending on strictness, you might want to raise an error here:
        # raise ValueError(f"NaNs found after fill: {nan_cols_after.index.tolist()}")
    else:
        print("NaNs handled successfully.")
        
    # --- 5. Data Splitting ---
    # As per your requirements:
    # Training: 2017-08-27 13:00:00 to 2023-12-31 23:00:00
    # Testing:  2024-01-01 00:00:00 to 2024-12-31 23:00:00 (or end of data if earlier)

    train_start_date = pd.to_datetime("2017-08-27 13:00:00")
    train_end_date = pd.to_datetime("2023-12-31 23:59:59") # Inclusive end for slicing
    test_start_date = pd.to_datetime("2024-01-01 00:00:00")
    # test_end_date can be the end of the DataFrame

    # Filter out data outside the global range if necessary, though splitting handles this.
    df_features = df_features[df_features.index >= train_start_date]

    train_df = df_features[df_features.index <= train_end_date]
    test_df = df_features[df_features.index >= test_start_date]

    print(f"Training data shape: {train_df.shape}, from {train_df.index.min()} to {train_df.index.max()}")
    print(f"Testing data shape: {test_df.shape}, from {test_df.index.min()} to {test_df.index.max()}")

    if train_df.empty:
        print("Warning: Training DataFrame is empty. Check date ranges and CSV data.")
    if test_df.empty:
        print("Warning: Testing DataFrame is empty. Check date ranges and CSV data.")

    return train_df, test_df, expected_features


if __name__ == '__main__':
    # Example usage:
    # Adjust this path relative to where data_manager.py is located,
    # or use an absolute path.
    # Assuming BTC_hourly_with_features.csv is in the 'btc_rl' directory,
    # and data_manager.py is in 'btc_rl/rl_trader/src/'
    
    # Construct the path to the CSV file relative to this script
    # This script is in .../src/
    # CSV is in .../ (one level up from rl_trader, not inside rl_trader)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    rl_trader_dir = os.path.dirname(current_script_dir)  # .../rl_trader/
    project_root = os.path.dirname(rl_trader_dir)        # .../btc_rl/
    csv_path = os.path.join(project_root, 'BTC_hourly_with_features.csv')
    csv_path = os.path.normpath(csv_path) # Cleans up path (e.g. /../../)

    # IMPORTANT: Verify 'timestamp_column_in_your_csv' is the correct name
    # of the date/time column in your CSV file.
    # Common names: 'timestamp', 'Date', 'Time', 'datetime', 'Open time'
    # If your CSV already has the date as index and it's read correctly by default,
    # you might need to adjust loading logic or ensure `pd.read_csv` uses `index_col=0, parse_dates=True`.
    # For this example, I'm assuming a column needs to be converted.
    date_col_in_csv = 'timestamp' # <--- !!! PLEASE VERIFY THIS MATCHES YOUR CSV !!!

    try:
        train_data, test_data, features = load_and_preprocess_data(csv_path, date_column_name=date_col_in_csv)

        print("\n--- Train Data Sample ---")
        print(train_data.head())
        print(f"Train data NaNs: {train_data.isnull().sum().sum()}")

        print("\n--- Test Data Sample ---")
        print(test_data.head())
        print(f"Test data NaNs: {test_data.isnull().sum().sum()}")
        
        print(f"\nFeatures being used ({len(features)}): {features}")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error during data processing: {e}")
    except Exception as e: # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}")