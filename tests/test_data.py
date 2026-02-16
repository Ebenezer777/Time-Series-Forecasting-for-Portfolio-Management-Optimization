import pandas as pd

def test_tsla_exists():
    df = pd.read_csv("data/processed/tsla.csv")
    assert not df.empty

def test_columns_exist():
    df = pd.read_csv("data/processed/spy.csv")
    assert "Close" in df.columns