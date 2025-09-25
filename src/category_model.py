# src/category_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def train_model(data_path="data/processed/transactions_mapped.csv", model_dir="models"):
    df = pd.read_csv(data_path)

    if "category_rule" not in df.columns:
        raise KeyError("❌ 'category_rule' column missing in dataset!")

    df = df[df['category_rule'] != 'Other']

    X = df['desc_clean']
    y = df['category_rule']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=500)),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "category_model.pkl")
    joblib.dump(pipeline, model_path)

    print(f"✅ Model saved at {model_path}")
    return pipeline

if __name__ == "__main__":
    train_model()
