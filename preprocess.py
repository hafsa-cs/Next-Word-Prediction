# ==========================================
# NEXT WORD PREDICTION SYSTEM
# Data Preprocessing & EDA
# ==========================================

import os
import re
import html
import string
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from wordcloud import WordCloud

from config import *


# ==========================================
# CREATE OUTPUT FOLDERS
# ==========================================

os.makedirs(GRAPH_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)


# ==========================================
# LOAD DATASET
# ==========================================

def load_dataset():
    print("=" * 50)
    print("Loading Dataset...")
    print("=" * 50)

    df = pd.read_csv(DATASET_PATH)

    print(f"Original Shape : {df.shape}")

    # 🔥 LIMIT TO MAX_ROWS (memory & speed control)
    if len(df) > MAX_ROWS:
        df = df.head(MAX_ROWS)
        print(f"Limited to first {MAX_ROWS} rows for training.")

    print(f"Final Shape    : {df.shape}")

    return df


# ==========================================
# TEXT CLEANING
# ==========================================

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Decode HTML
    text = html.unescape(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Keep only English letters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==========================================
# PREPROCESS DATASET
# ==========================================

def preprocess_dataset(df):
    print("\nCleaning Dataset...")

    # Keep only quote_text
    df = df[[TEXT_COLUMN]]

    # Rename column
    df.columns = ["text"]

    # Remove missing values
    df.dropna(inplace=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Clean text
    df["text"] = df["text"].apply(clean_text)

    # Remove empty rows
    df = df[df["text"].str.len() > 0]

    # Remove very short quotes
    df = df[df["text"].str.split().str.len() >= 4]

    print(f"Clean Dataset Shape : {df.shape}")

    return df


# ==========================================
# DATASET INFORMATION
# ==========================================

def dataset_statistics(df):
    print("\nDataset Statistics")
    print("-" * 30)

    total_quotes = len(df)
    total_words = sum(df["text"].str.split().apply(len))
    vocabulary = set()

    for sentence in df["text"]:
        vocabulary.update(sentence.split())

    avg_length = total_words / total_quotes if total_quotes else 0

    print("Total Quotes      :", total_quotes)
    print("Total Words       :", total_words)
    print("Vocabulary Size   :", len(vocabulary))
    print("Average Length    :", round(avg_length, 2))


# ==========================================
# TOP WORDS GRAPH
# ==========================================

def plot_top_words(df):
    print("\nGenerating Top Words Graph...")

    words = " ".join(df["text"]).split()
    counter = Counter(words)
    common = counter.most_common(20)

    labels = [x[0] for x in common]
    values = [x[1] for x in common]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.xticks(rotation=45)
    plt.title("Top 20 Most Frequent Words")
    plt.tight_layout()
    plt.savefig(f"{GRAPH_FOLDER}/top_words.png")
    plt.close()


# ==========================================
# WORD CLOUD
# ==========================================

def generate_wordcloud(df):
    print("Generating Word Cloud...")

    # Sample up to 5000 rows for the cloud (safe for large datasets)
    sample_n = min(5000, len(df))
    sample_text = df["text"].sample(n=sample_n, random_state=42)
    text = " ".join(sample_text)

    cloud = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        max_words=200
    ).generate(text)

    plt.figure(figsize=(12, 6))
    plt.imshow(cloud)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{GRAPH_FOLDER}/wordcloud.png")
    plt.close()


# ==========================================
# COMPLETE PIPELINE
# ==========================================

def prepare_data():
    df = load_dataset()
    df = preprocess_dataset(df)
    dataset_statistics(df)
    plot_top_words(df)
    generate_wordcloud(df)

    print("\nPreprocessing Completed Successfully.")
    return df


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    dataset = prepare_data()
    print(dataset.head())
