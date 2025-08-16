import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """Load insurance data and set appropriate data types."""
    df = pd.read_csv(file_path)
    
    # Convert TransactionMonth to datetime
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], format='%Y-%m')
    
    # Convert categorical columns
    categorical_cols = ['Province', 'Gender', 'VehicleType', 'Make', 'Model', 'CoverType']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    return df