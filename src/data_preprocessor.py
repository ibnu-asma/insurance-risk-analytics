import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer new features from the insurance dataset.
    
    
    Args:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    # Ensure TransactionMonth is datetime
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
    df['TransactionYear'] = df['TransactionMonth'].dt.year
    
    # Calculate LossRatio
    df['LossRatio'] = np.where(df['TotalPremium'] != 0, df['TotalClaims'] / df['TotalPremium'], np.nan)
    
    # Time-based features
    if 'TransactionMonth' in df.columns:
        df['TransactionMonthNum'] = df['TransactionMonth'].dt.month
        df['TransactionQuarter'] = df['TransactionMonth'].dt.quarter
    
    # Binary feature
    df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)
    
    # Aggregate features (e.g., average claims by make)
    make_claims = df.groupby('make')['TotalClaims'].mean().rename('AvgClaimsByMake')
    df = df.join(make_claims, on='make')
    
    return df

def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Preprocess the dataset for modeling.
    
    Args:
        df (pd.DataFrame): Input DataFrame with engineered features.
    
    Returns:
        tuple: Preprocessed DataFrame and fitted scaler.
    """
    # Handle negative values (cap at 0, assuming refunds are errors)
    df['TotalPremium'] = df['TotalPremium'].clip(lower=0)
    df['TotalClaims'] = df['TotalClaims'].clip(lower=0)
    
    # Save original values for reporting
    df['TotalPremium_Original'] = df['TotalPremium']
    df['TotalClaims_Original'] = df['TotalClaims']
    
    # Impute missing CustomValueEstimate with median
    if 'CustomValueEstimate' in df.columns:
        df['CustomValueEstimate'] = df['CustomValueEstimate'].fillna(df['CustomValueEstimate'].median())
    
    # Encoding categorical variables
    categorical_cols = ['Province', 'Gender', 'VehicleType', 'make', 'Model', 'CrossBorder']
    for col in categorical_cols:
        if col in df.columns:
            if df[col].nunique() < 10:  # Low cardinality
                df = pd.get_dummies(df, columns=[col], drop_first=True)
            else:  # High cardinality
                df[col + '_freq'] = df[col].map(df[col].value_counts() / len(df))
    
    # Select numerical columns for scaling
    numerical_cols = ['TotalPremium', 'TotalClaims', 'SumInsured', 'kilowatts', 'CustomValueEstimate', 'CapitalOutstanding']
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    
    # Scale numerical features
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df, scaler

def load_and_preprocess(csv_path: str) -> tuple[pd.DataFrame, StandardScaler]:
    """Load data and apply feature engineering and preprocessing.
    
    Args:
        csv_path (str): Path to the .csv file.
    
    Returns:
        tuple: Preprocessed DataFrame and fitted scaler.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
    df = engineer_features(df)
    df, scaler = preprocess_data(df)
    return df, scaler

if __name__ == '__main__':
    csv_path = 'data/MachineLearningRating_v3.csv'
    df, scaler = load_and_preprocess(csv_path)
    print("Preprocessed data shape:", df.shape)