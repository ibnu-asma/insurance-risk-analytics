import pandas as pd
import os

def clean_number(value):
    """Convert string numbers with comma decimals to float, handle invalid values."""
    if pd.isna(value) or value in ['', ' ', 'Not specified']:
        return pd.NA
    try:
        if isinstance(value, str):
            value = value.replace(',', '.')
        return float(value)
    except (ValueError, TypeError):
        return pd.NA

def get_dtype_dict():
    """Return dtype dictionary for all columns."""
    return {
        'UnderwrittenCoverID': 'int64',
        'PolicyID': 'int64',
        'TransactionMonth': 'object',  # Handle as datetime later
        'IsVATRegistered': 'category',
        'Citizenship': 'category',
        'LegalType': 'category',
        'Title': 'category',
        'Language': 'category',
        'Bank': 'category',
        'AccountType': 'category',
        'MaritalStatus': 'category',
        'Gender': 'category',
        'Country': 'category',
        'Province': 'category',
        'PostalCode': 'category',
        'MainCrestaZone': 'category',
        'SubCrestaZone': 'category',
        'ItemType': 'category',
        'mmcode': 'category',
        'VehicleType': 'category',
        'RegistrationYear': 'int32',
        'make': 'category',
        'Model': 'category',
        'Cylinders': 'float32',
        'cubiccapacity': 'float32',
        'kilowatts': 'float32',
        'bodytype': 'category',
        'NumberOfDoors': 'float32',
        'VehicleIntroDate': 'category',
        'CustomValueEstimate': 'float32',
        'AlarmImmobiliser': 'category',
        'TrackingDevice': 'category',
        'CapitalOutstanding': 'object',  # Clean later
        'NewVehicle': 'category',
        'WrittenOff': 'category',
        'Rebuilt': 'category',
        'Converted': 'category',
        'CrossBorder': 'category',
        'NumberOfVehiclesInFleet': 'float32',
        'SumInsured': 'float32',
        'TermFrequency': 'category',
        'CalculatedPremiumPerTerm': 'float32',
        'ExcessSelected': 'category',
        'CoverCategory': 'category',
        'CoverType': 'category',
        'CoverGroup': 'category',
        'Section': 'category',
        'Product': 'category',
        'StatutoryClass': 'category',
        'StatutoryRiskType': 'category',
        'TotalPremium': 'float32',
        'TotalClaims': 'float32'
    }

def convert_txt_to_csv(txt_path: str, csv_path: str, delimiter: str = '|') -> None:
    """Convert a pipe-separated .txt file to .csv format."""
    dtype_dict = get_dtype_dict()
    
    try:
        df = pd.read_csv(txt_path, delimiter=delimiter, encoding='utf-8', dtype=dtype_dict, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(txt_path, delimiter=delimiter, encoding='latin1', dtype=dtype_dict, low_memory=False)
    
    # Standardize categorical columns before saving
    categorical_cols = ['MaritalStatus', 'Gender', 'CrossBorder']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.add_categories(['']).fillna('')
    
    df.replace(['', 'Not specified'], pd.NA, inplace=True)
    
    if 'CapitalOutstanding' in df.columns:
        df['CapitalOutstanding'] = df['CapitalOutstanding'].apply(clean_number)
    
    df.to_csv(csv_path, index=False)
    print(f"Converted {txt_path} to {csv_path}")

def load_data(csv_path: str) -> pd.DataFrame:
    """Load insurance data from a .csv file and set appropriate data types."""
    dtype_dict = get_dtype_dict()
    
    df = pd.read_csv(csv_path, dtype=dtype_dict, low_memory=False)
    
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
    
    if 'CapitalOutstanding' in df.columns:
        df['CapitalOutstanding'] = df['CapitalOutstanding'].apply(clean_number)
    
    return df

if __name__ == '__main__':
    txt_path = 'data/MachineLearningRating_v3.txt'
    csv_path = 'data/MachineLearningRating_v3.csv'
    if os.path.exists(txt_path):
        convert_txt_to_csv(txt_path, csv_path)
    else:
        print(f"Error: {txt_path} not found")