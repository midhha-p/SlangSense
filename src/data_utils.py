
import re
import zipfile
from pathlib import Path

import pandas as pd


# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_datasets():
    """Load the three project datasets."""

    slang_path = DATA_DIR / "genz_slang.csv"
    tweets_path = DATA_DIR / "tweets.zip"
    trends_path = DATA_DIR / "Viral_Trends.csv"

    slang_df = pd.read_csv(slang_path)
    trends_df = pd.read_csv(trends_path)

    # Read the CSV stored inside the ZIP file
    with zipfile.ZipFile(tweets_path, "r") as zip_file:
        csv_files = [
            name for name in zip_file.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_files:
            raise ValueError("No CSV file found inside tweets.zip")

        with zip_file.open(csv_files[0]) as file:
            tweets_df = pd.read_csv(file)

    return slang_df, tweets_df, trends_df


def clean_text(text):
    """Clean text before NLP analysis."""

    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_slang_dictionary(slang_df):
    """Create a dictionary of slang terms and their meanings."""

    dictionary = {}

    for _, row in slang_df.iterrows():
        term = str(row.get("slang_term", "")).strip()
        meaning = str(row.get("meaning", "")).strip()

        if term and term.lower() != "nan":
            dictionary[term.lower()] = meaning

    return dictionary



def detect_slang(text, slang_dictionary):
    cleaned_text = clean_text(text)
    words = cleaned_text.split()
    found_terms = []

    for term, meaning in slang_dictionary.items():

        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"

        if not re.search(pattern, cleaned_text):
            continue

        positions = [
            i for i, word in enumerate(words)
            if word == term
        ]

        skip = False

        for pos in positions:

            nearby = words[max(0, pos - 3):pos + 4]

            # Ordinary use of tea as a beverage
            if term == "tea":
                beverage_context = {
                    "hot", "cold", "cup", "drink", "water",
                    "milk", "sugar", "brew", "sip", "kettle"
                }

                if any(word in beverage_context for word in nearby):
                    skip = True

            # Ordinary use of ate as eating food
            if term == "ate":
                food_context = {
                    "lunch", "dinner", "breakfast", "food",
                    "meal", "sandwich", "pizza", "rice",
                    "burger", "snack"
                }

                if any(word in food_context for word in nearby):
                    skip = True

        if not skip:
            found_terms.append({
                "term": term,
                "meaning": meaning
            })

    return found_terms
