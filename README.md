# SWYNEX AI — Academic & Campus Intelligence Suite

> **SWYNEX Technologies AI Internship — Task 1 (AI Problem Design), Task 2 (Model Integration), & Task 3 (Student Announcement Intelligence System)**  
> *A unified, production-grade end-to-end AI project suite demonstrating real-world problem formulation, dataset design, baseline benchmarking, TF-IDF machine learning classification, model evaluation, uncertainty guardrails, ambiguity detection, explainable AI, entity extraction, anomaly warnings, and a unified interactive web application.*

---

## 📌 Executive Summary

* **Project Name:** EduIntent AI & Student Announcement Intelligence System
* **Domain:** Natural Language Processing (NLP) / Higher Education Support Automation
* **Core Tasks Covered:**
  * **Task 1:** AI Problem Design & Metric Formulation
  * **Task 2:** Model Integration, API/Inference Engine & Real-time Web Application
  * **Task 3:** Student Announcement Intelligence System (Categorization, Non-Hallucinatory Extraction, Rule-Based Priority & Evaluation Lab)
* **Datasets:**
  * 450 balanced hand-curated student academic questions across 9 categories (`student_queries.csv`)
  * 175 balanced college announcements across 7 categories (`college_announcements.csv`)
  * 30 hand-annotated test cases spanning 6 test typologies for systematic evaluation (`evaluation_announcements.json`)
* **Task 1 & 2 Performance:** Calibrated Linear SVM achieving **87.78% Test Accuracy**, **0.8746 Macro F1**, **89.72% 5-Fold CV Accuracy**.
* **Task 3 Performance:** TF-IDF + Logistic Regression Classifier achieving **86.67% Accuracy**, **0.8202 Macro F1**, **100% Deadline Extraction F1**, **94.7% Event Date Extraction F1**, **78.1% Guardrail Detection Rate**.

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

---

## 6. Task 3: Student Announcement Intelligence System

Task 3 extends the system from reactive student inquiry handling to **proactive campus announcement intelligence**. College announcements arrive via disparate channels (WhatsApp, email, bulletin boards, LMS) causing missed deadlines and academic confusion.

### Key Capabilities & Architecture

1. **Multi-Class Announcement Categorization:**
   * 7 Core Classes: `Exams`, `Assignments`, `Events`, `Holidays`, `Placements`, `Fees`, `General`.
   * Model: Calibrated Logistic Regression with sublinear TF-IDF word & n-gram features (Achieved **86.67% Test Accuracy**, **0.8202 Macro F1**).
   * Edge-Case Safety: Vague or non-academic inputs route to `Unsupported / Low Confidence` guardrail.

2. **Strictly Non-Hallucinatory Information Extraction:**
   * **Deadlines:** Contextual regex engine anchored on deadline keywords (`before`, `by`, `deadline`, `submit by`, `last date`). Achieved **100% Precision & 100% Recall (F1: 100.0%)**.
   * **Event Dates:** Distinct extraction for event start/conduct dates (`held on`, `commence`, `on`, `during`). Achieved **94.7% F1**.
   * **Target Audience:** Automatic regex extraction of academic cohorts (e.g. `5th and 7th semester B.Tech`, `Final Year MCA`).
   * **Required Actions:** Extraction of imperative action verbs (`collect`, `submit`, `register`, `pay`, `upload`). Achieved **88.2% F1**.

3. **Transparent Rule-Based Priority Engine:**
   * Rather than treating urgency as a black box, priority is determined by explicit, inspectable rules:
     * **Urgent / High:** Impending deadlines, exam notices, fee dues, immediate action requirements.
     * **Medium:** Standard event notices, scheduled submissions with ample time.
     * **Low / Informational:** General circulars, holiday notices, elective orientations.
   * Full explainability provided with every priority assignment (lists exact contributing factors).

4. **Edge-Case & Anomaly Guardrails:**
   * **Missing Deadline Anomaly:** Detects actionable announcements (e.g. fee payments, exam registrations) that omit a deadline date and alerts administrators.
   * **Conflicting Dates Anomaly:** Detects logical anomalies where submission deadlines are scheduled *after* event conduct dates.
   * **Ambiguity Detection:** Flags cross-domain overlap (e.g., campus placement drive with an associated coding test).
   * **Input Sanitization & Validation:** Reject null, empty, symbols-only, or extreme character-count inputs with diagnostic feedback.

5. **Evaluation Lab & Benchmarking:**
   * 30 manually curated benchmark test cases spanning 6 test typologies:
     * `normal` (7 cases)
     * `missing_deadline` (5 cases)
     * `conflicting_dates` (4 cases)
     * `ambiguous` (5 cases)
     * `out_of_domain` (4 cases)
     * `malformed` (5 cases)
   * Detailed reports generated: [`reports/evaluation_report.md`](file:///c:/Users/Dell/Desktop/SWYNEX-AI-Problem-Design/reports/evaluation_report.md) and [`reports/failure_case_analysis.md`](file:///c:/Users/Dell/Desktop/SWYNEX-AI-Problem-Design/reports/failure_case_analysis.md).

---

## 7. Repository Folder Structure

```
SWYNEX-AI-Problem-Design/
├── app.py                          # Unified Multi-Tab Streamlit Dashboard (Tasks 1, 2 & 3)
├── create_dataset.py               # Task 1 & 2 student queries dataset generator
├── create_announcement_dataset.py  # Task 3 announcement dataset generator
├── create_evaluation_dataset.py    # Task 3 evaluation benchmark dataset generator
├── run_evaluation.py               # Task 3 Evaluation Lab benchmark runner
├── export_pdf_report.py            # PDF document report exporter
├── requirements.txt                # Python dependencies
├── README.md                       # Comprehensive Project Documentation
├── EduIntent_AI_Task1_Problem_Design.pdf # Published PDF submission document
│
├── dataset/
│   ├── student_queries.csv         # 450 balanced student queries (9 categories)
│   ├── college_announcements.csv   # 175 college announcements (7 categories)
│   └── evaluation_announcements.json # 30 hand-annotated benchmark evaluation samples
│
├── models/
│   ├── best_model.pkl              # Trained Calibrated Linear SVM (Student Queries)
│   ├── tfidf_vectorizer.pkl        # Saved TF-IDF vectorizer (Student Queries)
│   ├── model_metadata.json         # Saved metrics & categories (Task 1 & 2)
│   ├── announcement_classifier.pkl # Trained Classifier (Announcements)
│   ├── announcement_tfidf.pkl      # Saved TF-IDF vectorizer (Announcements)
│   └── announcement_metadata.json  # Saved metrics & categories (Task 3)
│
├── reports/
│   ├── confusion_matrix.png        # Task 1 & 2 Confusion Matrix plot
│   ├── model_comparison.png        # Task 1 & 2 Model Benchmark comparison plot
│   ├── announcement_confusion_matrix.png # Task 3 Confusion Matrix
│   ├── announcement_model_comparison.png # Task 3 Model comparison
│   ├── evaluation_report.md        # Comprehensive Task 3 Evaluation Report
│   ├── evaluation_results.json     # Machine-readable evaluation results
│   └── failure_case_analysis.md    # Systematic failure mode & post-mortem analysis
│
├── src/
│   ├── preprocessing.py            # Text cleaning and data splitting logic (Queries)
│   ├── train.py                    # Model training, evaluation & artifact exporter (Queries)
│   ├── predict.py                  # XAI, Ambiguity Detection & Inference Engine (Queries)
│   ├── announcement_analyzer.py    # Task 3 Core Engine (Classification + Extraction + Guardrails)
│   ├── announcement_model.py       # Task 3 Model training & artifact persistence
│   ├── announcement_ui.py          # Task 3 Streamlit UI components
│   ├── evaluation_lab.py           # Task 3 Benchmark evaluation suite & reporter
│   └── validation.py               # Input sanitization and bounds checking module
│
└── tests/
    ├── test_system.py              # Automated test suite for Task 1 & Task 2
    └── test_announcement_analyzer.py # Automated test suite for Task 3 (30 tests)
```

---

## 8. How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Automated Test Suites
Run all 30 unit tests across Task 1, 2, and 3:
```bash
python -m unittest discover tests
```

### 3. Run Task 3 Evaluation Lab Benchmark
Execute benchmark evaluation and regenerate evaluation reports & failure case post-mortems:
```bash
python run_evaluation.py
```

### 4. Launch the Unified Web Application (Tasks 1, 2 & 3)
```bash
streamlit run app.py
```
> 🌐 **Opens in your browser at `http://localhost:8501`**  
> Access all features directly from the unified sidebar:
> * 📢 **Smart Analyzer:** Real-time announcement parsing, date/entity extraction & priority estimation.
> * 🧪 **Evaluation Lab:** Live metric explorer, confusion matrix & interactive failure-case inspector.
> * 📋 **Example Library:** 1-click test scenarios across all edge-case typologies.
> * 📖 **Task 3 Specs & Rules:** Transparent rule tables and engineering design documentation.
> * 🏠 **Overview & Query Classifier:** Task 1 & Task 2 EduIntent student query system.

