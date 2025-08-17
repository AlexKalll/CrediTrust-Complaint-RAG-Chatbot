import pytest
import pandas as pd
from pathlib import Path
from src.data_preprocessing import load_data, clean_text, preprocess_data

@pytest.fixture
def sample_raw_dataframe():
    data = {
        'Complaint ID': [1, 2, 3, 4, 5, 6],
        'Product': ['Credit card', 'Personal loan', 'Credit card', 'Buy Now, Pay Later (BNPL)', 'Savings account', 'Credit card'],
        'Consumer complaint narrative': [
            'This is a test complaint about a credit card. It has some text.',
            'Another complaint regarding a personal loan. The terms were unclear.',
            'Third complaint, credit card issue. Billing error.',
            'BNPL service charged me incorrectly. Need a refund.',
            'Savings account interest rate is too low. Unhappy with the bank.',
            'Complaint with \n special characters and HTML &amp; entities. <br> Also, some numbers 12345.'
        ],
        'Issue': ['Billing error', 'Unclear terms', 'Interest rate', 'Incorrect charge', 'Low interest', 'Special characters'],
        'Company response to consumer': ['Closed with explanation', 'Closed with explanation', 'Closed with explanation', 'Closed with explanation', 'Closed with explanation', 'Closed with explanation'],
        'ZIP code': [12345, 67890, 11223, 33445, 55667, 99887]
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_csv_file(tmp_path, sample_raw_dataframe):
    file_path = tmp_path / "test_raw_complaints.csv"
    sample_raw_dataframe.to_csv(file_path, index=False)
    return file_path

def test_load_data(temp_csv_file, sample_raw_dataframe):
    df = load_data(temp_csv_file)
    pd.testing.assert_frame_equal(df, sample_raw_dataframe)
    with pytest.raises(FileNotFoundError):
        load_data(Path("non_existent_file.csv"))
    with pytest.raises(ValueError, match="CSV file is empty or does not contain expected columns."):
        empty_file = temp_csv_file.parent / "empty.csv"
        pd.DataFrame().to_csv(empty_file)
        load_data(empty_file)
    with pytest.raises(ValueError, match="Missing required columns in the CSV file."):
        missing_col_file = temp_csv_file.parent / "missing_col.csv"
        sample_raw_dataframe.drop(columns=['Consumer complaint narrative']).to_csv(missing_col_file, index=False)
        load_data(missing_col_file)

def test_clean_text():
    assert clean_text("Hello, World!") == "hello world"
    assert clean_text("  Leading and trailing spaces.  ") == "leading and trailing spaces"
    assert clean_text("Text with\nnewlines and\ttabs.") == "text with newlines and tabs"
    assert clean_text("Special characters: !@#$%^&*()_+") == "special characters"
    assert clean_text("Numbers 123 and symbols &amp; <br>") == "numbers and symbols"
    assert clean_text("Mixed Case String") == "mixed case string"
    assert clean_text(None) == ""
    assert clean_text(123) == "123"

def test_preprocess_data(sample_raw_dataframe):
    processed_df = preprocess_data(sample_raw_dataframe)
    assert 'Cleaned_Narrative' in processed_df.columns
    assert not processed_df['Cleaned_Narrative'].isnull().any()
    assert processed_df['Cleaned_Narrative'].dtype == 'object'

    # Check if text cleaning was applied
    expected_cleaned_narrative_part = [
        'this is a test complaint about a credit card it has some text',
        'another complaint regarding a personal loan the terms were unclear',
        'third complaint credit card issue billing error',
        'bnpl service charged me incorrectly need a refund',
        'savings account interest rate is too low unhappy with the bank',
        'complaint with special characters and html entities also some numbers'
    ]
    for i, expected_text in enumerate(expected_cleaned_narrative_part):
        assert processed_df.loc[i, 'Cleaned_Narrative'] == expected_text

    # Test with a DataFrame missing the 'Consumer complaint narrative' column
    with pytest.raises(ValueError, match="'Consumer complaint narrative' column not found"):
        preprocess_data(sample_raw_dataframe.drop(columns=['Consumer complaint narrative']))

    # Test with empty DataFrame
    empty_df = pd.DataFrame(columns=sample_raw_dataframe.columns)
    processed_empty_df = preprocess_data(empty_df)
    assert processed_empty_df.empty
    assert 'Cleaned_Narrative' in processed_empty_df.columns