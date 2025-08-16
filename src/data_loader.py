import pandas as pd
import os

def convert_txt_to_csv(txt_path: str, csv_path: str, delimiter: str = '\t') -> None:
    """Convert a .txt file to .csv format.
    
    Args:
        txt_path (str): Path to the input .txt file.
        csv_path (str): Path to save the output .csv file.
        delimiter (str): Delimiter for the .txt file (default: tab).
    """
    try:
        df = pd.read_csv(txt_path, delimiter=delimiter, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(txt_path, delimiter=delimiter, encoding='latin1')
    
    df.to_csv(csv_path, index=False)
    print(f"Converted {txt_path} to {csv_path}")

def load_data(csv_path: str) -> pd.DataFrame:
    """Load insurance data from a .csv file and set appropriate data types.
    
    Args:
        csv_path (str): Path to the .csv file.
    
    Returns:
        pd.DataFrame: Loaded and preprocessed DataFrame.
    """
    df = pd.read_csv(csv_path)
    
    # Convert TransactionMonth to datetime
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], format='%Y-%m', errors='coerce')
    
    # Convert categorical columns
    categorical_cols = [
        'Province', 'Gender', 'VehicleType', 'Make', 'Model', 'CoverType',
        'IsVATRegistered', 'Citizenship', 'LegalType', 'Title', 'Language',
        'Bank', 'AccountType', 'MaritalStatus', 'MainCrestaZone', 'SubCrestaZone'
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    return df

if __name__ == '__main__':
    # Convert .txt to .csv when script is run directly
    txt_path = 'data/MachineLearningRating_v3.txt'
    csv_path = 'data/MachineLearningRating_v3.csv'
    if os.path.exists(txt_path):
        convert_txt_to_csv(txt_path, csv_path)
    else:
        print(f"Error: {txt_path} not found")