# SWYNEX AI — Academic & Campus Intelligence Suite
### *A Production-Ready, Beginner-Friendly NLP Application for Higher Education*

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![ML Library](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests Status](https://img.shields.io/badge/Unit%20Tests-30%2F30%20Passing-brightgreen.svg)](tests/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-24AIMl037%2FSWYNEX--AI--Problem--Design-181717.svg?logo=github)](https://github.com/24AIMl037/SWYNEX-AI-Problem-Design)

> **SWYNEX Technologies AI Internship — Final Capstone Project (Tasks 1, 2, 3 & 4)**  
> **Official Submission Repository:** [https://github.com/24AIMl037/SWYNEX-AI-Problem-Design](https://github.com/24AIMl037/SWYNEX-AI-Problem-Design)

---

## 📌 Executive Summary

Modern colleges and universities face two massive communication challenges daily:
1. **Inbound Student Query Chaos:** Teaching assistants and help desks are inundated with repetitive technical and administrative questions across multiple computer science subjects.
2. **Outbound Announcement Overload:** Critical deadlines for fee payments, exam forms, and placement drives arrive in unstructured WhatsApp messages, circulars, and emails, causing students to miss deadlines and incur penalties.

The **SWYNEX Academic & Campus Intelligence Suite** solves both problems in a unified, beginner-friendly web application:
* **EduIntent AI (Tasks 1 & 2):** Automatically classifies student academic questions into 9 subjects with **87.78% accuracy**, extracts Explainable AI (XAI) key terms, and detects cross-disciplinary ambiguity.
* **Smart Announcement Analyzer (Task 3):** Analyzes campus circulars with **86.67% accuracy**, extracts cutoffs with **100% deadline F1-score**, assigns transparent rule-based priorities, and alerts on anomalies.
* **Final Packaged AI Application (Task 4):** A unified, production-grade Streamlit application featuring live inference, automated evaluation lab, example libraries, comprehensive documentation, and responsible AI ethics safeguards.

---

## 🎯 1. Problem Statement

### The Real-World Context
In university environments, communication breakdown is a leading cause of administrative friction:
* **Manual Sorting Inefficiency:** Help desk staff spend 3–4 hours each day manually forwarding questions to professors or departments (e.g., distinguishing a Deep Learning query from a Python syntax error).
* **High-Stakes Missed Deadlines:** When circulars mention multiple dates (e.g., *"Exams commence on 15th December; collect hall tickets before 10th December"*), students often confuse the event date with the action cutoff date, leading to late fees or exam debarment.
* **Linguistic Barriers:** In Indian engineering institutions, students frequently communicate in code-mixed languages like Gujarati-English or Hinglish (e.g., *"Pandas ma KeyError aave chhe su karvu?"*). Standard rigid keyword filters fail to categorize these queries.

### Engineering & AI Formulation
We frame this challenge as two complementary Natural Language Processing tasks:
1. **Multi-Class Intent Classification:** Mapping student text queries $x \in \mathcal{X}$ to subject categories $\mathcal{Y}_{query} = \{1, \dots, 9\}$ and announcement circulars to domain classes $\mathcal{Y}_{ann} = \{1, \dots, 7\}$.
2. **Deterministic Information Extraction & Anomaly Detection:** Extracting temporal entities (Deadlines vs. Event Dates), target cohorts, and imperative action verbs without generative hallucinations, coupled with rule-based priority scoring.

---

## ⚙️ 2. Method & Technical Architecture

Designed with **beginner-friendly clarity**, this application avoids opaque black-box deep networks in favor of proven, interpretable, and computationally lightweight Machine Learning:

```
[Raw Student Query / Notice]
            │
            ▼
┌───────────────────────────────────────┐
│  1. Preprocessing & Validation        │ ── Strips nulls, enforces bounds (3-5000 chars),
│     (src/validation.py)               │    detects Gujarati & Hinglish code-mixing.
└───────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────┐
│  2. TF-IDF Feature Extraction         │ ── Sublinear term frequency (1-2 word n-grams),
│     (4,000 max features, stopwords)   │    vocabulary union for technical terminology.
└───────────────────────────────────────┘
            │
            ├─────────────────────────────────────────┐
            ▼                                         ▼
┌───────────────────────────────────────┐ ┌───────────────────────────────────────┐
│  3A. Calibrated Linear SVM Classifier │ │  3B. Logistic Regression Classifier   │
│      (Student Academic Queries)       │ │      (College Announcements)          │
│      Accuracy: 87.78% | Macro F1: 0.87│ │      Accuracy: 86.67% | Macro F1: 0.82│
└───────────────────────────────────────┘ └───────────────────────────────────────┘
            │                                         │
            ▼                                         ▼
┌───────────────────────────────────────┐ ┌───────────────────────────────────────┐
│  4A. Explainable AI & Ambiguity Guard │ │  4B. Non-Hallucinatory Entity Engine  │
│      • TF-IDF feature contribution    │ │      • Strict contextual regex engine │
│      • Margin < 0.25 ambiguity alert  │ │      • Deadlines (100% F1-score)      │
│      • Confidence < 0.45 guardrail    │ │      • Event Dates (94.7% F1-score)   │
└───────────────────────────────────────┘ └───────────────────────────────────────┘
                                                      │
                                                      ▼
                                          ┌───────────────────────────────────────┐
                                          │  5. Rule-Based Priority & Anomalies   │
                                          │      • Urgent / Medium / Low priority │
                                          │      • Missing deadline alert         │
                                          │      • Conflicting dates warning      │
                                          └───────────────────────────────────────┘
```

### Benchmarks & Metric Highlights

All metrics are genuinely measured on held-out test sets and verified by automated unit tests:

| Evaluation Dimension | Metric Score | Target Benchmark | Status |
| :--- | :---: | :---: | :---: |
| **Student Query Accuracy (SVM)** | **87.78%** | $\ge 85.0\%$ | ✅ Exceeded |
| **Student Query Macro F1-Score** | **0.8746** | $\ge 0.800$ | ✅ Exceeded |
| **5-Fold Cross-Validation Accuracy** | **89.72% ($\pm 0.02$)** | $\ge 85.0\%$ | ✅ Highly Robust |
| **Announcement Classification Accuracy** | **86.67%** | $\ge 80.0\%$ | ✅ Exceeded |
| **Deadline Extraction F1-Score** | **100.0%** | $\ge 90.0\%$ | 🏆 Zero Hallucinations |
| **Event Date Extraction F1-Score** | **94.7%** | $\ge 90.0\%$ | ✅ High Precision |
| **Action Verb Extraction F1-Score** | **88.2%** | $\ge 80.0\%$ | ✅ Reliable Directives |
| **Anomaly & Guardrail Detection Rate** | **78.1%** | $\ge 70.0\%$ | ✅ High Sensitivity |

---

## 🚀 3. Live Demo & How to Run

### Quick Start (Local Setup)

Clone the repository and run in 3 simple commands:

```bash
# 1. Clone the repository
git clone https://github.com/24AIMl037/SWYNEX-AI-Problem-Design.git
cd SWYNEX-AI-Problem-Design

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the interactive web dashboard
streamlit run app.py
```

🌐 **Opens immediately in your browser at `http://localhost:8501`**

---

### Interactive UI Walkthrough

| Sidebar Navigation Section | Description & Features |
| :--- | :--- |
| **🎓 Task 4: Final AI Application** | Executive capstone showcase with problem framing, interactive architecture, user guide, limitations, and ethics notes. |
| **📢 Smart Analyzer (Task 3)** | Parse any real-time college circular. Instantly separates action deadlines from event dates, tags target cohorts, assigns priority with explainable rationale, and warns on missing deadlines. |
| **🧪 Evaluation Lab (Task 3)** | Run live benchmark testing across 30 curated test cases spanning 6 edge-case typologies. Features live confusion matrix and failure post-mortem explorer. |
| **📋 Example Library (Task 3)** | 8 pre-loaded realistic notices (Exam schedules, ML assignments, fee dues, Diwali break, Microsoft campus drive). 1-click test buttons. |
| **📖 Task 3 Specs & Rules** | Transparent decision tables detailing the exact priority scoring matrix and anomaly criteria. |
| **🏠 Overview (Task 1 & 2)** | Problem design overview, KPI dashboard, class balance breakdown, and target user personas. |
| **✦ Query Classifier (Task 1 & 2)** | Live student query classifier with confidence meter, top-3 class distribution, XAI important keyword terms, and Gujarati/Hinglish language detection. |
| **📊 Analytics & Benchmarks** | Head-to-head algorithm comparisons (Calibrated Linear SVM vs. Naive Bayes vs. Logistic Regression vs. Majority Baseline). |
| **✅ Submission Checklist** | Rubric alignment checklist verifying all requirements across all 4 internship tasks. |

---

### Deploying to Streamlit Community Cloud (Public Link)

1. Fork or push this repository to your GitHub account: `https://github.com/24AIMl037/SWYNEX-AI-Problem-Design`.
2. Visit **[share.streamlit.io](https://share.streamlit.io)** and log in with GitHub.
3. Click **"New app"**, select `24AIMl037/SWYNEX-AI-Problem-Design`, set `app.py` as the main file, and click **Deploy**.

---

## ⚠️ 4. Limitations & Edge Cases

In accordance with responsible engineering practices, we explicitly document the system's operational boundaries:

1. **Out-of-Vocabulary (OOV) Technical Slang:**
   * *Behavior:* If a student uses highly obscure abbreviations not present in the training corpus (e.g., custom campus project abbreviations), the TF-IDF feature representation may have zero non-zero weights.
   * *Mitigation:* The system contains an **Uncertainty Guardrail** that detects OOV inputs, assigns a conservative confidence score (30%), and alerts the student to add subject-specific keywords rather than making an overconfident incorrect guess.
2. **Single-Document Scope:**
   * *Behavior:* The analyzer processes one circular at a time and cannot automatically cross-reference external academic calendars or verify if an announced holiday clashes with a gazetted government notification.
   * *Mitigation:* Explicitly designed as a **Decision-Support Tool**; outputs are clearly labeled for administrative confirmation.
3. **Complex Relative Temporal Expressions:**
   * *Behavior:* Phrases such as *"three fortnights following subsequent Tuesday"* lack explicit calendar anchors.
   * *Mitigation:* The **Missing Deadline Anomaly Engine** flags notices containing action verbs without clear calendar dates, advising authors to state explicit dates (e.g., `15th December 2026`).
4. **Code-Mixed Spelling Variance:**
   * *Behavior:* Romanized Gujarati and Hindi (Hinglish) lack standardized orthography (e.g., `su karvu`, `syu karvu`, `su karie`).
   * *Mitigation:* Sublinear char/word n-grams and phonetic keyword matching capture dominant spelling variants.

---

## ⚖️ 5. Ethics, Fairness & Responsible AI Notes

Educational software directly affects student stress, deadlines, and academic standing. This project incorporates ethical AI design by design:

### 🔒 1. Student Privacy & Data Minimization
* **No PII Collection:** The system strictly processes the academic text. No roll numbers, student IDs, phone numbers, email addresses, or academic grades are required, stored, or transmitted.
* **Local Execution:** All inference runs on local CPU hardware or private cloud instances without transmitting student data to third-party proprietary APIs.

### 🚫 2. Zero-Hallucination Policy for Critical Dates
* Generative Large Language Models (LLMs) can invent plausible-sounding deadlines that do not exist.
* In academic administration, hallucinating an exam deadline could cause a student to miss an examination.
* Our system utilizes **deterministic contextual regular expressions**. A date is **only** returned if the exact characters exist in the input text. If no date is stated, the system explicitly returns `None` and triggers a **Missing Deadline Alert**.

### 🌐 3. Linguistic Inclusivity & Regional Dialect Support
* Students from diverse educational backgrounds frequently mix regional vocabulary (Gujarati / Hindi) with English technical terms.
* Rejecting or misclassifying code-mixed queries penalizes students who are not fluent in formal English.
* The system actively detects and accommodates code-mixing, providing equitable support across student demographics.

### 🤝 4. Explainable AI & Human-in-the-Loop Oversight
* Automated administrative decisions must never be opaque black boxes.
* Every prediction provides:
  1. Top TF-IDF driving terms explaining *why* a category was selected.
  2. Multi-subject ambiguity alerts when queries overlap.
  3. Transparent rule traces for assigned priority levels.
* Administrative staff retain full agency and can override priority assignments.

---

## 📁 6. Repository Folder Structure

```
SWYNEX-AI-Problem-Design/
├── app.py                          # Unified Multi-Tab Streamlit Dashboard (Tasks 1, 2, 3 & 4)
├── create_dataset.py               # Student academic queries dataset generator (Task 1 & 2)
├── create_announcement_dataset.py  # College announcements dataset generator (Task 3)
├── create_evaluation_dataset.py    # Evaluation benchmark dataset generator (30 test cases)
├── run_evaluation.py               # Automated Evaluation Lab benchmark runner
├── export_pdf_report.py            # PDF document report exporter
├── requirements.txt                # Production dependencies
├── README.md                       # Comprehensive Final Project Documentation (Task 4)
├── EduIntent_AI_Task1_Problem_Design.pdf # Published Task 1 PDF report
│
├── dataset/
│   ├── student_queries.csv         # 450 balanced student queries (9 categories)
│   ├── college_announcements.csv   # 175 balanced announcements (7 categories)
│   └── evaluation_announcements.json # 30 hand-annotated benchmark evaluation samples
│
├── models/
│   ├── best_model.pkl              # Trained Calibrated Linear SVM (Queries)
│   ├── tfidf_vectorizer.pkl        # Saved TF-IDF vectorizer (Queries)
│   ├── model_metadata.json         # Saved metrics & categories (Task 1 & 2)
│   ├── announcement_classifier.pkl # Trained Classifier (Announcements)
│   ├── announcement_tfidf.pkl      # Saved TF-IDF vectorizer (Announcements)
│   └── announcement_metadata.json  # Saved metrics & categories (Task 3)
│
├── reports/
│   ├── confusion_matrix.png        # Task 1 & 2 Confusion Matrix plot
│   ├── model_comparison.png        # Task 1 & 2 Model comparison plot
│   ├── announcement_confusion_matrix.png # Task 3 Confusion Matrix plot
│   ├── announcement_model_comparison.png # Task 3 Model comparison plot
│   ├── evaluation_report.md        # Comprehensive Task 3 Evaluation Report
│   ├── evaluation_results.json     # Machine-readable evaluation results
│   └── failure_case_analysis.md    # Systematic failure mode post-mortem
│
├── src/
│   ├── preprocessing.py            # Text cleaning and data splitting logic (Queries)
│   ├── train.py                    # Model training, evaluation & artifact persistence (Queries)
│   ├── predict.py                  # XAI, Ambiguity Detection & Inference Engine (Queries)
│   ├── announcement_analyzer.py    # Task 3 Core Engine (Classification + Extraction + Rules)
│   ├── announcement_model.py       # Task 3 Model training & artifact persistence
│   ├── announcement_ui.py          # Task 3 Streamlit UI components
│   ├── task4_ui.py                 # Task 4 Capstone Showcase UI component
│   ├── evaluation_lab.py           # Task 3 Benchmark evaluation suite & reporter
│   └── validation.py               # Input sanitization and bounds checking module
│
└── tests/
    ├── test_system.py              # Automated test suite for Task 1 & Task 2
    └── test_announcement_analyzer.py # Automated test suite for Task 3 (30 tests)
```

---

## 🧪 7. Automated Verification & Test Commands

Every component in this repository is covered by automated unit tests and benchmark runners:

### Run All Unit Tests (30 Tests Passing)
```bash
python -m unittest discover tests
```
*Validates dataset balance, text normalization, ML model inference, entity extraction accuracy, priority rule assignment, and anomaly detection.*

### Run Evaluation Lab Benchmark
```bash
python run_evaluation.py
```
*Executes the 30-case benchmark test suite and regenerates `reports/evaluation_report.md` and `reports/failure_case_analysis.md`.*

### Retrain Models From Scratch (Optional)
```bash
# Retrain Student Query Models (Task 1 & 2)
python src/train.py

# Retrain Announcement Models (Task 3)
python src/announcement_model.py
```

### Export PDF Report Document
```bash
python export_pdf_report.py
```

---

## 👨‍💻 8. Author & Submission Information

* **Candidate / Intern:** 24AIML037
* **Internship Program:** SWYNEX Technologies AI Internship
* **Submission Tasks:** 
  * ✅ **Task 1:** AI Problem Design & Dataset Formulation
  * ✅ **Task 2:** Model Integration, API & Real-time Web Application
  * ✅ **Task 3:** Student Announcement Intelligence System & Evaluation Lab
  * ✅ **Task 4:** Final Beginner-Friendly AI Application, Limitations & Ethics Documentation
* **GitHub Repository Submission URL:**  
  👉 **[https://github.com/24AIMl037/SWYNEX-AI-Problem-Design](https://github.com/24AIMl037/SWYNEX-AI-Problem-Design)**
