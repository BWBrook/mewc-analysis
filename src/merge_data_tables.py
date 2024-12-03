import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from common import load_config

def validate_directory(directory_path):
    """
    Validate that the given path is an existing directory.
    """
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        print(f"Error: The directory '{dir_path}' does not exist or is not a directory.")
        sys.exit(1)
    return dir_path

def get_csv_files(directory_path):
    """
    Retrieve all .csv files in the given directory.
    """
    csv_files = list(directory_path.glob("*.csv"))
    return csv_files

def validate_csv(file_path, required_columns):
    """
    Validate that the CSV file contains required columns with valid data.
    """
    try:
        df = pd.read_csv(file_path, dtype=str)
        
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Rejected '{file_path.name}': Missing columns {missing_columns}.")
            return False
                
        # Check for missing values in required columns
        for col in required_columns:
            if df[col].isna().any() or (df[col] == '').any():
                print(f"Rejected '{file_path.name}': Contains missing values in '{col}' column.")
                return False
                    
        return True
    except Exception as e:
        print(f"Rejected '{file_path.name}': Failed to read CSV. Error: {e}")
        return False

def merge_dataframes(csv_files):
    """
    Merge multiple CSV files into a single DataFrame with a 'source' column.
    """
    data_frames = []
    merged_files = []
    for file_path in tqdm(csv_files, desc="Merging CSV files", unit="file"):
        try:
            df = pd.read_csv(file_path, dtype=str)
            df['source'] = file_path.stem
            
            # Convert timestamp to datetime, keeping as datetime object
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
                if df['timestamp'].isna().any():
                    print(f"Warning: Found NA timestamps in {file_path.name}")
                    print("Sample problematic values:", df[df['timestamp'].isna()]['timestamp'].head())
            
            data_frames.append(df)
            merged_files.append(file_path.name)
        except Exception as e:
            print(f"Warning: Failed to read '{file_path.name}'. Error: {e}")
    
    if not data_frames:
        return pd.DataFrame()
    
    # Merge all dataframes
    combined_df = pd.concat(data_frames, ignore_index=True, sort=False)
    
    # Handle non-timestamp columns for NA replacement
    cols_to_clean = [col for col in combined_df.columns if col not in ['camera_site', 'timestamp']]
    for col in cols_to_clean:
        combined_df[col] = combined_df[col].replace(['', 'None', None], 'NA')
    
    print("\nSuccessfully merged the following MEWC data tables:")
    for file in merged_files:
        print(f"  - {file}")
    
    return combined_df

def reorder_columns(df):
    """
    Reorder the columns of the DataFrame to start with 'camera_site', 'class_name', 'timestamp'
    and end with 'source'. Other columns remain in between in any order.
    """
    key_columns = ['camera_site', 'class_name', 'timestamp']
    end_column = ['source']
    other_columns = [col for col in df.columns if col not in key_columns + end_column]
    
    ordered_columns = key_columns + other_columns + end_column
    df = df.reindex(columns=ordered_columns)
    return df

def main():
    config = load_config()

    # Retrieve and validate 'data_tables' directory
    data_tables_dir = config.get("data_tables")
    if not data_tables_dir:
        print("Error: 'data_tables' parameter is missing in 'params.yaml'.")
        sys.exit(1)

    data_tables_path = validate_directory(data_tables_dir)

    # Retrieve all .csv files in the directory
    csv_files = get_csv_files(data_tables_path)

    if not csv_files:
        print(f"No .csv files found in '{data_tables_path}'. Ending script.")
        sys.exit(0)

    required_columns = ['camera_site', 'class_name', 'timestamp']
    valid_csv_files = []

    print("Validating CSV files...")
    for file_path in tqdm(csv_files, desc="Checking CSV files", unit="file"):
        if validate_csv(file_path, required_columns):
            valid_csv_files.append(file_path)

    if not valid_csv_files:
        print("No valid CSV files found after validation. Ending script.")
        sys.exit(0)

    print(f"Found {len(valid_csv_files)} valid CSV file(s) to merge.")

    # Merge the valid CSV files
    combined_df = merge_dataframes(valid_csv_files)

    if combined_df.empty:
        print("No data to merge after processing CSV files. Ending script.")
        sys.exit(0)

    # Reorder the columns
    combined_df = reorder_columns(combined_df)

    # Sort the combined dataframe
    sort_columns = ['camera_site', 'timestamp']
    missing_sort_cols = [col for col in sort_columns if col not in combined_df.columns]
    if missing_sort_cols:
        print(f"Error: Missing columns for sorting: {missing_sort_cols}. Cannot sort the DataFrame.")
        sys.exit(1)

    print("Sorting the combined DataFrame...")
    try:
        combined_df = combined_df.sort_values(by=sort_columns)
        # Convert timestamps to string format only after sorting
        combined_df['timestamp'] = combined_df['timestamp'].dt.strftime('%d/%m/%Y %H:%M:%S')
    except Exception as e:
        print(f"Error while sorting the DataFrame: {e}")
        sys.exit(1)

    # Define the output file path
    output_file = data_tables_path / "merged_data_table.csv"

    # Save the combined dataframe
    try:
        combined_df.to_csv(output_file, index=False)
        print(f"Merged DataFrame saved successfully to '{output_file}'.")
    except Exception as e:
        print(f"Error saving the merged DataFrame: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
