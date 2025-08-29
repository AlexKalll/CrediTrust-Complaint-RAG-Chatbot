import pandas as pd
import re
import string
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data (only once)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The absolute path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the CSV file is empty or missing required columns.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found at {file_path}")

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("CSV file is empty or does not contain expected columns.")

    required_columns = ["Consumer complaint narrative", "Product", "Complaint ID"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(
            "Missing required columns in the CSV file. Expected: 'Consumer complaint narrative', 'Product', 'Complaint ID'"
        )

    # Add Issue column if it doesn't exist
    if "Issue" not in df.columns:
        df["Issue"] = "General complaint"

    return df


def clean_text(text: str) -> str:
    """
    Cleans a single string of text by lowercasing, removing punctuation, numbers,
    and non-alphabetic words, and lemmatizing.

    Args:
        text (str): The input text string.

    Returns:
        str: The cleaned and lemmatized text string.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()  # Lowercasing and ensure it's a string
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces

    tokens = word_tokenize(text)
    lemmatized_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and word.isalpha()
    ]
    return " ".join(lemmatized_tokens)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all preprocessing steps to the 'Consumer complaint narrative' column in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with the preprocessed text column added as 'Cleaned_Narrative'.

    Raises:
        ValueError: If the 'Consumer complaint narrative' column is not found.
    """
    if "Consumer complaint narrative" not in df.columns:
        raise ValueError(
            "'Consumer complaint narrative' column not found in the DataFrame."
        )

    # Fill NaN values in 'Consumer complaint narrative' with empty string before processing
    df["Consumer complaint narrative"] = df["Consumer complaint narrative"].fillna("")

    df["Cleaned_Narrative"] = df["Consumer complaint narrative"].apply(clean_text)
    return df


if __name__ == "__main__":
    # Example Usage (for testing purposes)
    # Create a dummy CSV file for demonstration
    dummy_data = {
        "Complaint ID": [1, 2, 3, 4],
        "Product": ["Credit card", "Personal loan", "Credit card", "Bank account"],
        "Consumer complaint narrative": [
            "This is a test complaint about a credit card. It has numbers like 123 and symbols!@#.",
            "Another complaint regarding bank services. Missing values here.",
            "A third complaint, very clear and concise. No issues.",
            None,  # Example of a missing value
        ],
        "Issue": ["Billing error", "Unclear terms", "No issues", "Missing info"],
        "Company response to consumer": ["Closed", "Closed", "Closed", "Closed"],
        "ZIP code": [12345, 67890, 11223, 45678],
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_file_path = Path("dummy_complaints.csv")
    dummy_df.to_csv(dummy_file_path, index=False)

    try:
        # Load data
        df = load_data(dummy_file_path)
        print("Original DataFrame:")
        print(df)

        # Preprocess the 'Consumer complaint narrative' column
        processed_df = preprocess_data(df.copy())
        print("\nProcessed DataFrame:")
        print(processed_df)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up dummy file
        if dummy_file_path.exists():
            dummy_file_path.unlink()
