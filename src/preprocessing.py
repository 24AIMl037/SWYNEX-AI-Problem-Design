import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(text: str) -> str:
    """
    Preprocesses and cleans query text:
    1. Converts to lowercase
    2. Removes punctuation & special symbols
    3. Normalizes whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_and_preprocess_data(file_path: str):
    """
    Loads dataset from CSV, cleans text queries, and returns dataframe with cleaned queries.
    """
    df = pd.read_csv(file_path)
    df["Cleaned_Question"] = df["Question"].apply(clean_text)
    return df

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Splits dataset into stratified train and test sets.
    """
    X = df["Cleaned_Question"]
    y = df["Category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
