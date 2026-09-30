import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

from preprocessing import load_and_preprocess_data, split_data

def train_and_evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "dataset", "student_queries.csv")
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports")
    results_dir = os.path.join(base_dir, "results")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("Loading data...")
    df = load_and_preprocess_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # TF-IDF Feature Extraction
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Models dictionary
    models = {
        "Baseline (Majority Class)": DummyClassifier(strategy="most_frequent"),
        "Naive Bayes (MultinomialNB)": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=500, random_state=42),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42))
    }

    results = {}
    best_model_name = None
    best_f1 = -1.0
    best_model_obj = None

    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)

        acc = accuracy_score(y_test, preds)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)

        # 5-fold cross-validation on train set
        cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5, scoring="accuracy")

        results[name] = {
            "Accuracy": float(acc),
            "Precision (Macro)": float(p_macro),
            "Recall (Macro)": float(r_macro),
            "F1-Score (Macro)": float(f1_macro),
            "F1-Score (Weighted)": float(f1_weighted),
            "CV_Mean_Accuracy": float(np.mean(cv_scores)),
            "CV_Std_Accuracy": float(np.std(cv_scores))
        }

        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro F1-Score: {f1_macro:.4f}")
        print(f"5-Fold CV Mean Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_model_name = name
            best_model_obj = model

    print(f"\n======================================")
    print(f"Best Performing Model: {best_model_name} with Macro F1 = {best_f1:.4f}")
    print(f"======================================\n")

    # Detailed classification report for best model
    best_preds = best_model_obj.predict(X_test_vec)
    cls_report_dict = classification_report(y_test, best_preds, output_dict=True)
    cls_report_str = classification_report(y_test, best_preds)
    print("Classification Report (Best Model):\n", cls_report_str)

    # Perform Qualitative Error Analysis on Test Set
    test_indices = y_test.index
    df_test_errors = []
    for idx, actual_cat, pred_cat in zip(test_indices, y_test, best_preds):
        if actual_cat != pred_cat:
            q_text = df.loc[idx, "Question"]
            reason = "Keyword Overlap / Ambiguous Vocabulary"
            if (actual_cat in ["Machine Learning", "Deep Learning"]) and (pred_cat in ["Machine Learning", "Deep Learning"]):
                reason = "Shared ML/DL terminology (loss functions, optimization, neural nets)"
            elif (actual_cat in ["DBMS", "DAA / Algorithms"]) and (pred_cat in ["DBMS", "DAA / Algorithms"]):
                reason = "Shared indexing / data structure terminology (Trees, Graphs, Indexing)"
            elif (actual_cat in ["Python / Data Science", "Machine Learning"]):
                reason = "Python Data Science library usage in ML tasks"

            df_test_errors.append({
                "Question": q_text,
                "Expected": actual_cat,
                "Predicted": pred_cat,
                "Reason": reason
            })

    # Save artifacts into models/
    joblib.dump(vectorizer, os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(best_model_obj, os.path.join(models_dir, "best_model.pkl"))

    categories = sorted(list(df["Category"].unique()))
    meta = {
        "best_model_name": best_model_name,
        "categories": categories,
        "results_summary": results,
        "classification_report": cls_report_dict,
        "error_analysis": df_test_errors
    }
    with open(os.path.join(models_dir, "model_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Save evaluation text to results/evaluation.txt
    eval_txt_path = os.path.join(results_dir, "evaluation.txt")
    with open(eval_txt_path, "w", encoding="utf-8") as f:
        f.write("====================================================\n")
        f.write("   EduIntent AI – Model Evaluation & Benchmark Summary   \n")
        f.write("====================================================\n\n")
        f.write(f"Best Performing Model: {best_model_name}\n")
        f.write(f"Test Accuracy: {results[best_model_name]['Accuracy']*100:.2f}%\n")
        f.write(f"Macro F1-Score: {results[best_model_name]['F1-Score (Macro)']:.4f}\n")
        f.write(f"5-Fold CV Mean Accuracy: {results[best_model_name]['CV_Mean_Accuracy']*100:.2f}%\n\n")
        f.write("--- Benchmark Comparison Across Algorithms ---\n")
        for m, metrics in results.items():
            f.write(f"Model: {m:<30} | Accuracy: {metrics['Accuracy']*100:6.2f}% | Macro F1: {metrics['F1-Score (Macro)']:6.4f} | 5-Fold CV: {metrics['CV_Mean_Accuracy']*100:6.2f}%\n")
        f.write("\n--- Classification Report ---\n")
        f.write(cls_report_str)
        f.write("\n--- Qualitative Error Analysis (Misclassified Test Cases) ---\n")
        if df_test_errors:
            for err in df_test_errors:
                f.write(f"Question: '{err['Question']}'\n")
                f.write(f"  -> Expected: {err['Expected']} | Predicted: {err['Predicted']}\n")
                f.write(f"  -> Reason: {err['Reason']}\n\n")
        else:
            f.write("No misclassifications recorded on test set.\n")

    # Plot Confusion Matrix
    labels = sorted(list(set(y_test)))
    cm = confusion_matrix(y_test, best_preds, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {best_model_name}", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Category", fontsize=12)
    plt.ylabel("Actual Category", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "confusion_matrix.png"), dpi=300)
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # Plot Model Comparison Bar Chart
    df_results = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    plt.figure(figsize=(10, 6))
    x_indices = np.arange(len(df_results))
    width = 0.35

    plt.bar(x_indices - width/2, df_results["Accuracy"], width, label="Accuracy", color="#2b5c8f")
    plt.bar(x_indices + width/2, df_results["F1-Score (Macro)"], width, label="Macro F1-Score", color="#46a094")
    plt.xticks(x_indices, df_results["Model"], rotation=15, ha="right", fontsize=10)
    plt.ylabel("Score", fontsize=12)
    plt.title("Model Comparison (Accuracy vs Macro F1-Score)", fontsize=14, fontweight="bold")
    plt.ylim(0, 1.1)
    plt.axhline(0.85, color="red", linestyle="--", label="Target Accuracy (85%)")
    plt.axhline(0.80, color="orange", linestyle=":", label="Target F1 (0.80)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "model_comparison.png"), dpi=300)
    plt.savefig(os.path.join(results_dir, "model_comparison.png"), dpi=300)
    plt.close()

    print(f"Artifacts saved to 'models/', 'reports/', and 'results/' successfully.")

if __name__ == "__main__":
    train_and_evaluate()
