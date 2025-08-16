import pandas as pd
import os

def clean_number(value):
    """Convert string numbers with comma decimals to float, handle invalid values.
    
    Args:
        value: Input value (string or other).
    
    Returns:
        float or pd.NA: Cleaned float value or NA for invalid inputs.
    """
    if pd.isna(value) or value in ['', ' ', 'Not specified']:
        return pd.NA
    try:
        # Replace comma with period and convert to float
        if isinstance(value, str):
            value = value.replace(',', '.')
        return float(value)
    except (ValueError, TypeError):
        return pd.NA

def convert_txt_to_csv(txt_path: str, csv_path: str, delimiter: str = '|') -> None:
    """Convert a pipe-separated .txt file to .csv format.
    
    Args:
        txt_path (str): Path to the input .txt file.
        csv_path (str): Path to save the output .csv file.
        delimiter (str): Delimiter for the .txt file (default: pipe).
    """
    # Read with CapitalOutstanding as string to avoid casting errors
    dtypes = {
        'CapitalOutstanding': 'object',  # Column 32
        'CrossBorder': 'category'       # Column 37
    }
    
    try:
        df = pd.read_csv(txt_path, delimiter=delimiter, encoding='utf-8', dtype=dtypes, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(txt_path, delimiter=delimiter, encoding='latin1', dtype=dtypes, low_memory=False)
    
    # Replace empty strings and placeholders with NaN
    df.replace(['', ' ', 'Not specified'], pd.NA, inplace=True)
    
    # Clean CapitalOutstanding
    if 'CapitalOutstanding' in df.columns:
        df['CapitalOutstanding'] = df['CapitalOutstanding'].apply(clean_number)
    
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
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
    
    # Convert categorical columns
    categorical_cols = [
        'Province', 'Gender', 'VehicleType', 'make', 'Model', 'CoverType',
        'IsVATRegistered', 'Citizenship', 'LegalType', 'Title', 'Language',
        'Bank', 'AccountType', 'MaritalStatus', 'MainCrestaZone', 'SubCrestaZone',
        'CrossBorder'
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Ensure numerical columns are float
    numerical_cols = ['CapitalOutstanding', 'TotalPremium', 'TotalClaims', 'SumInsured', 'kilowatts', 'CustomValueEstimate']
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

if __name__ == '__main__':
    txt_path = 'data/MachineLearningRating_v3.txt'
    csv_path = 'data/MachineLearningRating_v3.csv'
    if os.path.exists(txt_path):
        convert_txt_to_csv(txt_path, csv_path)
    else:
        print(f"Error: {txt_path} not found")