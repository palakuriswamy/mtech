"""
Multi-Algorithm Training Script for Fake Job Posting Detection
Trains: LSTM (Deep Learning) + Logistic Regression + Random Forest (TF-IDF)
"""
import os
import re
import pickle
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def download_dataset():
    """Download dataset if not present"""
    if os.path.exists("fake_job_postings.csv"):
        print("Dataset found locally.")
        return True
    url = "https://raw.githubusercontent.com/abbylmm/fake_job_posting/main/data/fake_job_postings.csv"
    try:
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, "fake_job_postings.csv")
        print("Dataset downloaded.")
        return True
    except Exception as e:
        print(f"Could not download dataset: {e}")
        return False

def clean_text(text):
    text = str(text) if pd.notna(text) else ""
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return text.lower().strip()

def main():
    if not download_dataset():
        print("Please add fake_job_postings.csv to run training.")
        return

    print("\n=== Loading Data ===")
    df = pd.read_csv("fake_job_postings.csv")
    
    # Fill NaN and create combined text
    text_cols = ['title', 'location', 'salary_range', 'company_profile', 'description', 
                 'requirements', 'benefits', 'employment_type', 'required_experience', 
                 'required_education', 'industry', 'function', 'department']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('')
    
    df['combined_text'] = df[text_cols].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    df['combined_text'] = df['combined_text'].apply(clean_text)
    y = df['fraudulent'].values

    print(f"Dataset: {len(df)} samples, {y.sum()} fraudulent")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['combined_text'], y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n=== Training TF-IDF Vectorizer ===")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    X_train_bal, y_train_bal = X_train_tfidf, y_train

    # Logistic Regression (class_weight handles imbalance)
    print("\n=== Training Logistic Regression ===")
    lr_model = LogisticRegression(C=1.0, max_iter=500, class_weight='balanced', random_state=42)
    lr_model.fit(X_train_bal, y_train_bal)
    lr_pred = lr_model.predict(X_test_tfidf)
    print(f"LR F1: {f1_score(y_test, lr_pred):.4f}")

    # Random Forest
    print("\n=== Training Random Forest ===")
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', 
                                      random_state=42, n_jobs=-1)
    rf_model.fit(X_train_bal, y_train_bal)
    rf_pred = rf_model.predict(X_test_tfidf)
    print(f"RF F1: {f1_score(y_test, rf_pred):.4f}")

    # Save models
    print("\n=== Saving Models ===")
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open("logistic_model.pkl", "wb") as f:
        pickle.dump(lr_model, f)
    with open("random_forest_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)

    print("\n=== Training Complete ===")
    print("Saved: tfidf_vectorizer.pkl, logistic_model.pkl, random_forest_model.pkl")
    print("\nClassification Report (Ensemble - majority vote):")
    ensemble_pred = (np.array(lr_pred) + np.array(rf_pred)) >= 1
    print(classification_report(y_test, ensemble_pred, target_names=['Legitimate', 'Fraudulent']))

if __name__ == "__main__":
    main()
