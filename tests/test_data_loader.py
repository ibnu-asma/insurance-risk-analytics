import pandas as pd
import pytest
import os
from src.data_loader import convert_txt_to_csv, load_data

@pytest.fixture
def sample_txt(tmp_path):
    """Create a sample .txt file for testing."""
    data = (
        "TransactionMonth|TotalPremium|TotalClaims|Province|CapitalOutstanding|MaritalStatus|Gender|CrossBorder\n"
        "2014-02-01 00:00:00|100|0|Gauteng|50000,00|Married|Male|\n"
        "2014-03-01 00:00:00|200|50|Western Cape|285700,00|Not specified||Yes\n"
        "2014-04-01 00:00:00|150|0|KwaZulu-Natal|N/A||Female|"
    )
    file_path = tmp_path / 'test_data.txt'
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(data)
    return file_path

@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample .csv file path for testing."""
    csv_path = tmp_path / 'test_data.csv'
    return csv_path

def test_convert_txt_to_csv(sample_txt, sample_csv):
    """Test .txt to .csv conversion."""
    convert_txt_to_csv(sample_txt, sample_csv, delimiter='|')
    assert os.path.exists(sample_csv)
    df = pd.read_csv(sample_csv)
    assert len(df) == 3
    assert pd.isna(df['CapitalOutstanding'].iloc[2])
    assert df['CapitalOutstanding'].iloc[0] == 50000.0
    assert df['MaritalStatus'].iloc[0] == 'Married'
    assert pd.isna(df['Gender'].iloc[1])

def test_load_data(sample_txt, sample_csv):
    """Test load_data function."""
    convert_txt_to_csv(sample_txt, sample_csv, delimiter='|')
    df = load_data(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert df['TransactionMonth'].dtype == 'datetime64[ns]'
    assert df['Province'].dtype.name == 'category'
    assert df['CapitalOutstanding'].dtype == 'float64'
    assert df['MaritalStatus'].dtype.name == 'category'
    assert df['Gender'].dtype.name == 'category'
    assert df['CrossBorder'].dtype.name == 'category'
    assert df['CapitalOutstanding'].iloc[1] == 285700.0