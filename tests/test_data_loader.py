import pandas as pd
import pytest
import os
from src.data_loader import convert_txt_to_csv, load_data

@pytest.fixture
def sample_txt(tmp_path):
    """Create a sample .txt file for testing."""
    data = (
        "TransactionMonth\tTotalPremium\tTotalClaims\tProvince\n"
        "2014-02\t100\t0\tGauteng\n"
        "2014-03\t200\t50\tWestern Cape"
    )
    file_path = tmp_path / 'test_data.txt'
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(data)
    return file_path

@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample .csv file for testing."""
    csv_path = tmp_path / 'test_data.csv'
    return csv_path

def test_convert_txt_to_csv(sample_txt, sample_csv):
    """Test .txt to .csv conversion."""
    convert_txt_to_csv(sample_txt, sample_csv)
    assert os.path.exists(sample_csv)
    df = pd.read_csv(sample_csv)
    assert len(df) == 2
    assert 'Province' in df.columns

def test_load_data(sample_txt, sample_csv):
    """Test load_data function."""
    convert_txt_to_csv(sample_txt, sample_csv)
    df = load_data(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert df['TransactionMonth'].dtype == 'datetime64[ns]'
    assert df['Province'].dtype.name == 'category'