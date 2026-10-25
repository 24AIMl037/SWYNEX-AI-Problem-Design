"""
announcement_model.py
---------------------
Training and serialization pipeline for the Student Announcement Intelligence System.
Trains TF-IDF + Logistic Regression and candidate classifiers on the 7-class
College Announcement Dataset:
['Exams', 'Assignments', 'Events', 'Holidays', 'Placements', 'Fees', 'General'].

Generates:
- models/announcement_classifier.pkl
- models/announcement_tfidf.pkl
- models/announcement_metadata.json
- reports/announcement_confusion_matrix.png
- reports/announcement_model_comparison.png
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

# Ensure src preprocessing is accessible
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, CURRENT_DIR)

from preprocessing import clean_text

def train_announcement_models(data_path: str = None, models_dir: str = None, reports_dir: str = None):
    if data_path is None:
        data_path = os.path.join(PROJECT_ROOT, "dataset", "college_announcements.csv")
    if models_dir is None:
        models_dir = os.path.join(PROJECT_ROOT, "models")
    if reports_dir is None:
        reports_dir = os.path.join(PROJECT_ROOT, "reports")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print(f"[1/5] Loading announcement dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Total samples: {len(df)}, Classes: {df['Category'].nunique()}")

    # Clean text
    df["Cleaned"] = df["Announcement"].astype(str).apply(clean_text)

    X = df["Cleaned"]
    y = df["Category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[2/5] Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # TF-IDF Feature Extraction with English stop words removal
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=4000,
        sublinear_tf=True,
        stop_words="english",
        min_df=1
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"TF-IDF Vocabulary size: {len(vectorizer.get_feature_names_out())}")

    # Benchmark models
    candidate_models = {
        "Baseline (Majority Class)": DummyClassifier(strategy="most_frequent"),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(C=2.0, max_iter=1000, random_state=42),
        "Linear SVM (Calibrated)": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42, dual=False))
    }

    results = {}
    best_name = "Logistic Regression" # default target
    best_f1 = -1.0
    fitted_models = {}

    print("[3/5] Benchmarking candidate models with 5-Fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in candidate_models.items():
        clf.fit(X_train_vec, y_train)
        preds = clf.predict(X_test_vec)

        acc = accuracy_score(y_test, preds)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)

        cv_scores = cross_val_score(clf, X_train_vec, y_train, cv=skf, scoring="accuracy")

        results[name] = {
            "Accuracy": float(acc),
            "Precision (Macro)": float(p_macro),
            "Recall (Macro)": float(r_macro),
            "F1-Score (Macro)": float(f1_macro),
            "F1-Score (Weighted)": float(f1_weighted),
            "CV_Mean_Accuracy": float(np.mean(cv_scores)),
            "CV_Std_Accuracy": float(np.std(cv_scores))
        }
        fitted_models[name] = clf

        print(f" -> {name:28s} | Test Acc: {acc*100:.2f}% | Macro F1: {f1_macro:.4f} | 5-Fold CV: {np.mean(cv_scores)*100:.2f}% (±{np.std(cv_scores)*100:.2f}%)")

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_name = name

    # Primary model is Logistic Regression as requested by Task 2 specification
    primary_clf = fitted_models["Logistic Regression"]
    primary_preds = primary_clf.predict(X_test_vec)

    categories = sorted(list(df["Category"].unique()))
    cls_report = classification_report(y_test, primary_preds, target_names=categories, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, primary_preds, labels=categories)

    print(f"\n[4/5] Primary Model: Logistic Regression")
    print(f"Test Accuracy: {results['Logistic Regression']['Accuracy']*100:.2f}%")
    print(f"Macro F1: {results['Logistic Regression']['F1-Score (Macro)']:.4f}")
    print(classification_report(y_test, primary_preds, target_names=categories, zero_division=0))

    # Save artifacts
    print("[5/5] Saving model artifacts and evaluation plots...")
    clf_path = os.path.join(models_dir, "announcement_classifier.pkl")
    vec_path = os.path.join(models_dir, "announcement_tfidf.pkl")
    meta_path = os.path.join(models_dir, "announcement_metadata.json")

    joblib.dump(primary_clf, clf_path)
    joblib.dump(vectorizer, vec_path)

    metadata = {
        "project": "Student Announcement Intelligence System",
        "task": "SWYNEX AI Internship - Task 3: Smart Announcement Analyzer",
        "primary_model": "Logistic Regression",
        "feature_extraction": "TF-IDF Vectorizer (ngram=(1,2), sublinear_tf=True, max_features=4000)",
        "categories": categories,
        "num_categories": len(categories),
        "total_dataset_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "vocabulary_size": len(vectorizer.get_feature_names_out()),
        "benchmark_results": results,
        "classification_report": cls_report,
        "confusion_matrix": cm.tolist()
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)
    plt.title("Confusion Matrix — College Announcement Classifier (Logistic Regression)", fontsize=12, pad=12)
    plt.xlabel("Predicted Category", fontsize=10)
    plt.ylabel("Actual Category", fontsize=10)
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    cm_plot_path = os.path.join(reports_dir, "announcement_confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()

    # Plot Model Comparison
    models_list = list(results.keys())
    acc_scores = [results[m]["Accuracy"] * 100 for m in models_list]
    f1_scores = [results[m]["F1-Score (Macro)"] * 100 for m in models_list]

    x = np.arange(len(models_list))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width/2, acc_scores, width, label="Accuracy (%)", color="#3b82f6")
    plt.bar(x + width/2, f1_scores, width, label="Macro F1 (%)", color="#10b981")
    plt.xticks(x, [m.replace(" (", "\n(") for m in models_list], fontsize=9)
    plt.ylabel("Score (%)", fontsize=10)
    plt.title("Model Benchmark Comparison on College Announcements", fontsize=12, pad=12)
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    comp_plot_path = os.path.join(reports_dir, "announcement_model_comparison.png")
    plt.savefig(comp_plot_path, dpi=200)
    plt.close()

    print(f"Successfully trained and saved:")
    print(f" - Model: {clf_path}")
    print(f" - Vectorizer: {vec_path}")
    print(f" - Metadata: {meta_path}")
    print(f" - CM Plot: {cm_plot_path}")
    print(f" - Comparison Plot: {comp_plot_path}")

    return metadata

if __name__ == "__main__":
    train_announcement_models()
