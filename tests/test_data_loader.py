import pandas as pd
import pytest
from src.data_loader import load_data

@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV for testing."""
    data = {
        'TransactionMonth': ['2014-02', '2014-03'],
        'TotalPremium': [100, 200],
        'TotalClaims': [0, 50],
        'Province': ['Gauteng', 'Western Cape']
    }
    file_path = tmp_path / 'test_data.csv'
    pd.DataFrame(data).to_csv(file_path, index=False)
    return file_path

def test_load_data(sample_csv):
    """Test load_data function."""
    df = load_data(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert df['TransactionMonth'].dtype == 'datetime64[ns]'
    assert df['Province'].dtype.name == 'category'