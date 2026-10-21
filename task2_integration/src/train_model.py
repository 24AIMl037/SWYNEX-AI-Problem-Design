"""
EduIntent AI - Model Training and Evaluation Pipeline
SWYNEX Technologies AI Internship - Task 2: Model Integration
Author: SWYNEX AI Intern

This script:
1. Loads and validates the student academic query dataset
2. Performs stratified train-test splitting (80% train, 20% test)
3. Evaluates and benchmarks multiple ML classifiers (Naive Bayes, Logistic Regression, Linear SVM, Random Forest)
4. Evaluates real metrics (Accuracy, Precision, Recall, F1)
5. Generates confusion matrix and model comparison visualization artifacts
6. Serializes the production TF-IDF Vectorizer and Logistic Regression Classifier
7. Saves comprehensive model metadata for the Streamlit dashboard
"""

import os
import sys
import json
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Add current directory to path to import local preprocessing
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, CURRENT_DIR)

from preprocessing import clean_text


def load_dataset(data_path: str):
    """Loads and validates the student query dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    
    # Normalize column names
    col_map = {col: col.strip().lower() for col in df.columns}
    df.rename(columns=col_map, inplace=True)

    if "question" not in df.columns or "category" not in df.columns:
        raise ValueError(f"Dataset must contain 'Question' and 'Category' columns. Found: {df.columns.tolist()}")

    df.dropna(subset=["question", "category"], inplace=True)
    df["clean_question"] = df["question"].apply(clean_text)
    # Filter empty queries
    df = df[df["clean_question"].str.len() > 0].reset_index(drop=True)

    print(f" Loaded {len(df)} samples across {df['category'].nunique()} academic categories.")
    print("\nClass Distribution:")
    for cat, count in df["category"].value_counts().items():
        print(f"  - {cat:<25}: {count} queries")

    return df


def train_and_evaluate(data_path: str = None, output_dir: str = None):
    """Main training, benchmarking, and serialization pipeline."""
    if data_path is None:
        data_path = os.path.join(PROJECT_ROOT, "data", "student_queries.csv")
    if output_dir is None:
        output_dir = PROJECT_ROOT

    models_dir = os.path.join(output_dir, "models")
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 65)
    print(" EduIntent AI - Model Integration Training Pipeline")
    print("=" * 65)

    # 1. Load dataset
    df = load_dataset(data_path)
    X = df["clean_question"]
    y = df["category"]
    categories = sorted(y.unique().tolist())

    # 2. Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"\n Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 3. TF-IDF Feature Extraction
    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        max_features=5000
    )
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    vocab_size = len(tfidf.vocabulary_)
    print(f" TF-IDF Vocabulary Size: {vocab_size} tokens (unigrams + bigrams)")

    # 4. Multi-Model Benchmark Comparison
    candidate_models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(C=2.0, max_iter=1000, random_state=42),
        "Linear SVM (Calibrated)": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42, dual=False)),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    benchmark_results = {}
    best_model_name = "Logistic Regression"  # Recommended by project plan

    print("\n Evaluating Candidate AI/ML Models:")
    print("-" * 65)
    print(f"{'Model Name':<28} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<10}")
    print("-" * 65)

    for name, clf in candidate_models.items():
        clf.fit(X_train_vec, y_train)
        y_pred = clf.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        benchmark_results[name] = {
            "accuracy": round(float(acc), 4),
            "precision_weighted": round(float(prec), 4),
            "recall_weighted": round(float(rec), 4),
            "f1_macro": round(float(f1_macro), 4),
            "f1_weighted": round(float(f1_weighted), 4)
        }
        print(f"{name:<28} | {acc * 100:>8.2f}% | {f1_macro * 100:>8.2f}% | {f1_weighted * 100:>8.2f}%")

    print("-" * 65)

    # 5. Primary Production Model: Logistic Regression
    primary_model = candidate_models[best_model_name]
    y_test_pred = primary_model.predict(X_test_vec)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_accuracy = accuracy_score(y_train, primary_model.predict(X_train_vec))
    class_report_str = classification_report(y_test, y_test_pred, digits=4)
    class_report_dict = classification_report(y_test, y_test_pred, output_dict=True, digits=4)

    print(f"\n Selected Production Model: {best_model_name}")
    print(f" Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f" Test Accuracy:  {test_accuracy * 100:.2f}%")
    print(f" Weighted F1:    {class_report_dict['weighted avg']['f1-score'] * 100:.2f}%\n")
    print("Detailed Classification Report on Unseen Test Set:")
    print(class_report_str)

    # Save Classification Report
    report_file = os.path.join(results_dir, "classification_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("EduIntent AI - Model Classification Report (Unseen Test Set)\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {best_model_name} with TF-IDF Vectorizer\n")
        f.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n\n")
        f.write(class_report_str)

    # 6. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_test_pred, labels=categories)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=categories,
        yticklabels=categories,
        cbar=True
    )
    plt.title(f"EduIntent AI - Test Set Confusion Matrix\n({best_model_name} | Accuracy: {test_accuracy*100:.1f}%)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Category", fontsize=11, fontweight="bold")
    plt.ylabel("True Category", fontsize=11, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f" Saved Confusion Matrix to {cm_path}")

    # 7. Model Comparison Bar Chart
    plt.figure(figsize=(10, 5.5))
    model_names = list(benchmark_results.keys())
    accuracies = [benchmark_results[m]["accuracy"] * 100 for m in model_names]
    f1_scores = [benchmark_results[m]["f1_macro"] * 100 for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='#1E88E5')
    rects2 = ax.bar(x + width/2, f1_scores, width, label='Macro F1 (%)', color='#26A69A')

    ax.set_ylabel('Score (%)', fontsize=11, fontweight="bold")
    ax.set_title('EduIntent AI - Model Benchmarking Comparison', fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10, rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()
    cmp_path = os.path.join(results_dir, "model_comparison.png")
    plt.savefig(cmp_path, dpi=200)
    plt.close()
    print(f" Saved Model Comparison Chart to {cmp_path}")

    # 8. Serialize Trained Artifacts
    tfidf_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    classifier_path = os.path.join(models_dir, "classifier.pkl")

    joblib.dump(tfidf, tfidf_path)
    joblib.dump(primary_model, classifier_path)
    print(f" Serialized TF-IDF Vectorizer to {tfidf_path}")
    print(f" Serialized Classifier to {classifier_path}")

    # 9. Save Metadata JSON
    metadata = {
        "project": "EduIntent AI - Intelligent Student Query Classification",
        "task": "SWYNEX AI Internship - Task 2: Model Integration",
        "primary_model": best_model_name,
        "feature_extraction": "TF-IDF Vectorizer (ngram=(1,2), sublinear_tf=True, max_features=5000)",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_dataset_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "vocabulary_size": vocab_size,
        "num_categories": len(categories),
        "categories": categories,
        "train_accuracy": round(float(train_accuracy), 4),
        "test_accuracy": round(float(test_accuracy), 4),
        "weighted_f1": round(float(class_report_dict['weighted avg']['f1-score']), 4),
        "macro_f1": round(float(class_report_dict['macro avg']['f1-score']), 4),
        "benchmark_comparison": benchmark_results,
        "per_class_metrics": {
            cat: {
                "precision": round(class_report_dict[cat]["precision"], 4),
                "recall": round(class_report_dict[cat]["recall"], 4),
                "f1_score": round(class_report_dict[cat]["f1-score"], 4),
                "support": int(class_report_dict[cat]["support"])
            }
            for cat in categories if cat in class_report_dict
        }
    }

    meta_path = os.path.join(models_dir, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f" Saved Model Metadata to {meta_path}")

    results_json = os.path.join(results_dir, "evaluation_metrics.json")
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n Training Pipeline Completed Successfully!")
    return metadata


if __name__ == "__main__":
    train_and_evaluate()
