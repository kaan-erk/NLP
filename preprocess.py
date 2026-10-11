# ==============================================================================
# STEP 2: DATA PREPROCESSING AND SPLITTING
# ==============================================================================
# This script:
# 1. Reads 'news_dataset_4000.csv' (Politics, Finance, Weather, Sport).
# 2. Generates 'clean_text' (lowercase, removed punctuation/numbers/stopwords).
#    NOTE: The raw 'text' column is preserved for BERT because Transformer
#    embeddings leverage capitalization and punctuation for context.
# 3. Ensures a strictly balanced class distribution across all categories.
# 4. Performs a stratified train-test split (80% Train, 20% Test).
# ==============================================================================

import pandas as pd
import re
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords

# Download NLTK English stopwords
nltk.download('stopwords', quiet=True)
ENGLISH_STOP_WORDS = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans raw news headlines for classical machine learning (TF-IDF).
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Keep only letters (remove punctuation and numbers)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # 3. Split into words
    words = text.split()

    # 4. Filter out stopwords and single-character tokens
    clean_words = []
    for word in words:
        if word not in ENGLISH_STOP_WORDS and len(word) > 1:
            clean_words.append(word)

    return " ".join(clean_words)

if __name__ == "__main__":
    print("==================================================")
    print("           DATA PREPROCESSING STARTED             ")
    print("==================================================")

    # 1. Load the scraped dataset
    dataset_file = "news_dataset_4000.csv"
    try:
        df = pd.read_csv(dataset_file)
        print(f"-> '{dataset_file}' loaded. Total records: {len(df)}")
    except FileNotFoundError:
        print(f"ERROR: '{dataset_file}' not found! Run scraper.py first.")
        exit()

    # 2. Clean text ('clean_text' for TF-IDF, preserve 'text' for BERT)
    print("-> Cleaning headlines (generating 'clean_text', preserving raw 'text')...")
    df['clean_text'] = df['text'].apply(clean_text)

    # Filter out empty or excessively short texts (fewer than 3 words)
    df = df[df['clean_text'].str.split().str.len() >= 3].copy()

    # 3. Equalize category counts for strict balance
    min_samples = df['label'].value_counts().min()
    print(f"\n-> Equalizing categories ({min_samples} samples per category)...")
    df = df.groupby('label').head(min_samples).reset_index(drop=True)

    print("\nCategory Distribution (After Preprocessing):")
    print(df['label'].value_counts())

    # 4. Stratified Train-Test Split (80% Train, 20% Test)
    print("\n-> Splitting data into 80% Train and 20% Test (Stratified)...")
    train_df, test_df = train_test_split(
        df, 
        test_size=0.20, 
        random_state=42, 
        stratify=df['label']
    )

    # 5. Save preprocessed splits
    train_df.to_csv("train_data.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv("test_data.csv", index=False, encoding="utf-8-sig")

    print("\n==================================================")
    print("              PROCESSING COMPLETED                ")
    print("==================================================")
    print(f"Training Data (Train) : {len(train_df)} rows (train_data.csv)")
    print(f"Testing Data (Test)   : {len(test_df)} rows (test_data.csv)")
    print("\nTraining Set Distribution:")
    print(train_df['label'].value_counts())
    print("\nTesting Set Distribution:")
    print(test_df['label'].value_counts())
