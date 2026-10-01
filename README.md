# EduIntent AI – Intelligent Student Query Classification System

> **Task 1 – AI Problem Design & Practical Implementation**  
> *A practical, measurable, explainable end-to-end AI project demonstrating real-world problem formulation, dataset design, baseline benchmarking, TF-IDF machine learning classification, model evaluation, uncertainty warnings, ambiguity detection, and an interactive web demo.*

---

## 📌 Executive Summary

* **Project Name:** EduIntent AI – Intelligent Student Query Classification System
* **Domain:** Natural Language Processing (NLP) / Academic Support Automation
* **AI Task:** Multi-Class Text Intent Classification
* **Dataset Size:** 450 balanced hand-curated student questions across 9 categories.
* **Target Accuracy:** $\ge 85\%$ *(Achieved: **87.78%**)*
* **Target Macro F1-Score:** $\ge 0.80$ *(Achieved: **0.8746**)*
* **5-Fold Cross-Validation Accuracy:** **89.72% ($\pm 0.02$)**

---

## ⭐ Unique AI System Features

Rather than simply building a basic classifier that outputs a single label, **EduIntent AI** incorporates mature real-world AI system design principles:

### 1. Explainable AI (XAI) – *"Why did AI choose this category?"*
Instead of showing only `Prediction: Deep Learning`, EduIntent AI extracts and displays the top TF-IDF key terms in the student's question that drove the classification (e.g. `Important terms: CNN, neural network, layers`).

### 2. Ambiguity Detection – Multi-Subject Overlap Handling
Real-world student queries are often cross-disciplinary. For queries like *"How can I implement CNN using Python?"*, the system detects probability overlap between `Deep Learning` and `Python / Data Science` and alerts the user:
> 🔀 **Ambiguous Multi-Subject Query Detected:** Primary: `Deep Learning` | Related: `Python / Data Science`

### 3. Low Confidence Warning – Uncertainty Guardrails
For vague inputs like *"Can you explain this?"* or out-of-vocabulary queries, the model refuses to output an overconfident wrong label. Instead, it displays:
> ⚠️ **Low-Confidence Warning (< 45%):** Please refine your question with more subject-specific context.

### 4. Language & Code-Mixing Detection (Gujarati + English / Hinglish)
Detects romanized code-mixed student queries (e.g. *"Pandas ma KeyError aave chhe su karvu?"*) and displays the detected language badge (`Gujarati / English Code-Mixed`).

---

## 1. Problem Statement

### English
Students frequently ask academic and technical questions related to core computer science and engineering subjects such as **Python, Data Science, Machine Learning, Deep Learning, DBMS, Operating Systems, Computer Networks, Flutter, DAA/Algorithms, and General Academic Administrative queries**.

When a large volume of queries is received by teaching assistants, college help desks, or online learning portals, manually reviewing and assigning every question to the correct subject category is time-consuming and inefficient.

The proposed solution is **EduIntent AI**, an intelligent text-classification system that automatically analyzes the content of a student's question, assigns it to the most relevant academic category, displays confidence scores, extracts explainability terms, and detects ambiguous queries.

---





---

## 2. Target Users

* **College Students:** Quick automated categorization and routing of technical queries.
* **Academic Support Teams & TAs:** Reduces manual ticket sorting workload.
* **College Help Desks & Administrative Counters:** Fast handling of student administrative inquiries.
* **Online EdTech Learning Platforms:** Automated tagging for student discussion forums.

---

## 3. Dataset Architecture & Source

* **Dataset Size:** 450 balanced, hand-curated academic queries (50 questions per category).
* **Location:** [`dataset/student_queries.csv`](file:///c:/Users/Dell/Desktop/SWYNEX-AI-Problem-Design/dataset/student_queries.csv)
* **Categories (9 Subjects):**
  1. `Python / Data Science`
  2. `Machine Learning`
  3. `Deep Learning`
  4. `DBMS`
  5. `Operating Systems`
  6. `Computer Networks`
  7. `Flutter`
  8. `DAA / Algorithms`
  9. `General Academic Query`
* **Train/Test Split:** Stratified 80% Training ($N=360$) and 20% Testing ($N=90$).

---

## 4. Evaluation Results & Benchmarks

To ensure rigorous evaluation, models were compared against a **Majority Class Baseline** on the held-out test set ($N=90$):

| Model Algorithm | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | 5-Fold CV Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Majority Class)** | 11.11% | 0.0123 | 0.1111 | 0.0222 | 11.11% ($\pm 0.00$) |
| **Naive Bayes (MultinomialNB)** | 85.56% | 0.8640 | 0.8556 | 0.8547 | 87.50% ($\pm 0.03$) |
| **Logistic Regression** | 83.33% | 0.8350 | 0.8333 | 0.8275 | 86.39% ($\pm 0.04$) |
| **Linear SVM (Calibrated)** 🏆 | **87.78%** | **0.8870** | **0.8778** | **0.8746** | **89.72% ($\pm 0.02$)** |

> ✅ **Success Criteria Check:**
> * Target Accuracy $\ge 85\%$ $\rightarrow$ **Achieved 87.78%**
> * Target Macro F1 $\ge 0.80$ $\rightarrow$ **Achieved 0.8746**
> * Significant gain over Baseline Dummy Model ($11.11\% \rightarrow 87.78\%$).

---

## 5. Qualitative Error Analysis

Evaluating why the model makes mistakes on specific test instances:

| Question | Expected Category | Predicted Category | Reason for Misclassification |
| :--- | :--- | :--- | :--- |
| *"Explain neural network optimization loss function"* | Deep Learning | Machine Learning | Shared ML/DL vocabulary (`neural network`, `optimization`) |
| *"What is B+ Tree structure in file indexing?"* | DBMS | DAA / Algorithms | Shared data structure & indexing terminology |
| *"How to plot correlation matrix in Pandas?"* | Python / Data Science | Python / Data Science | ✅ Correctly classified (High precision) |

---

## 6. Repository Folder Structure

```
SWYNEX-AI-Problem-Design/
├── app.py                      # Interactive Streamlit Web Application (with XAI & Ambiguity UI)
├── create_dataset.py           # Dataset generation script
├── create_notebook.py          # Jupyter notebook generation script
├── export_pdf_report.py        # PDF document report exporter
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive Task 1 Project Documentation
├── EduIntent_AI_Task1_Problem_Design.pdf # Published PDF submission document
├── dataset/
│   └── student_queries.csv     # 450 balanced student queries (9 categories)
├── models/
│   ├── best_model.pkl          # Trained Calibrated Linear SVM model
│   ├── tfidf_vectorizer.pkl    # Saved TF-IDF vectorizer
│   └── model_metadata.json     # Saved metrics, categories & error analysis
├── notebooks/
│   └── EduIntent_AI_Problem_Design.ipynb # Executable Jupyter Notebook
├── reports/
│   ├── confusion_matrix.png    # Saved Confusion Matrix plot
│   └── model_comparison.png    # Saved Model Benchmark comparison plot
├── results/
│   ├── confusion_matrix.png    # Evaluation Confusion Matrix image
│   ├── model_comparison.png    # Evaluation Benchmark chart
│   └── evaluation.txt          # Exported evaluation summary text
└── src/
    ├── preprocessing.py        # Text cleaning and data splitting logic
    ├── train.py                # Model training, evaluation & artifact exporter
    ├── predict.py              # XAI, Ambiguity Detection & Inference Engine
    └── model.py                # Model helper module
```

---

## 7. How to Run locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models & Export Results
```bash
python src/train.py
```

### 3. Run Inference Test with Unique Features
```bash
python src/predict.py
```

### 4. Launch Interactive Streamlit Web Demo
```bash
streamlit run app.py
```
*Access the web app in your browser at `http://localhost:8501`.*
