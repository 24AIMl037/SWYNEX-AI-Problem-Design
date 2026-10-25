# SWYNEX Technologies AI Internship — Task 3 Evaluation Report
**Project:** Student Announcement Intelligence System (Smart Announcement Analyzer)  
**Evaluation Date:** Real execution on 30 manually annotated test cases  
**Engine:** Scikit-Learn TF-IDF + Logistic Regression + Rule-Based Entity & Priority Engine  

---

## 1. Executive Summary & Headline Metrics

| Metric Dimension | Measured Score | Evaluation Benchmark Status |
| :--- | :---: | :---: |
| **Overall Classification Accuracy** | **86.67%** | ✅ Target Exceeded ($\ge 80\%$) |
| **Macro F1-Score** | **0.8202** | ✅ Balanced across all domains |
| **Weighted F1-Score** | **0.8641** | ✅ Robust against class skew |
| **Deadline Extraction F1-Score** | **100.0%** | Precision: 100.0% | Recall: 100.0% |
| **Event Date Extraction F1-Score** | **94.7%** | Precision: 90.0% | Recall: 100.0% |
| **Action Extraction F1-Score** | **88.2%** | Precision: 100.0% | Recall: 78.9% |
| **Guardrail & Anomaly Detection Rate** | **78.1%** | High sensitivity for edge cases |

---

## 2. Performance Breakdown by Announcement Test Typology

| Test Category Typology | Total Samples | Category Accuracy | Average Confidence | Description & Characteristics |
| :--- | :---: | :---: | :---: | :--- |
| **Ambiguous** | 5 | 100.0% | 41.2% | Evaluates ambiguous behavior and guardrails |
| **Conflicting Dates** | 4 | 75.0% | 51.8% | Evaluates conflicting_dates behavior and guardrails |
| **Malformed** | 5 | 100.0% | 0.0% | Evaluates malformed behavior and guardrails |
| **Missing Deadline** | 5 | 100.0% | 37.8% | Evaluates missing_deadline behavior and guardrails |
| **Normal** | 7 | 100.0% | 54.9% | Evaluates normal behavior and guardrails |
| **Out Of Domain** | 4 | 100.0% | 0.0% | Evaluates out_of_domain behavior and guardrails |

---

## 3. Confusion Matrix Analysis

*Classes evaluated:* `Assignments, Events, Exams, Fees, General, Holidays, Placements, Unsupported / Low Confidence`

```text
Confusion Matrix (Rows = Ground Truth, Columns = Predicted):
                              Assignments  Events  Exams  Fees  General  Holidays  Placements  Unsupported / Low Confidence
Assignments                             5       0      0     0        0         0           0                             0
Events                                  0       2      0     0        0         0           1                             0
Exams                                   0       0      2     1        0         1           0                             0
Fees                                    0       0      0     3        0         0           0                             0
General                                 0       0      0     0        2         0           0                             0
Holidays                                0       0      0     0        0         1           0                             0
Placements                              0       0      1     0        0         0           2                             0
Unsupported / Low Confidence            0       0      0     0        0         0           0                             9
```

---

## 4. Entity & Information Extraction Evaluation

Extraction performance evaluated strictly against ground truth:
* **Deadlines:** Precision: `100.0%`, Recall: `100.0%`, F1: `100.0%`
* **Event Dates:** Precision: `90.0%`, Recall: `100.0%`, F1: `94.7%`
* **Required Actions:** Precision: `100.0%`, Recall: `78.9%`, F1: `88.2%`

---

## 5. Recorded Failure Cases & Actionable Recommendations

Total failure cases identified: **5** out of 30 tests.

### Failure Case 1: [AMBIG-01] (ambiguous)
* **Announcement Text:** *"Students with backlogs must pay the supplementary examination fee of Rs. 500 and submit the re-exam registration form before 18th November 2026."*
* **Expected Category:** `Exams`
* **Predicted Category:** `Fees`
* **Failure Reasons:** Missed expected anomaly flag: ['ambiguous_category']
* **Root Cause:** Vocabulary overlap between competing domains (e.g. 'Exams' and 'Fees'). TF-IDF weights were balanced across both sets of terms.
* **Suggested Improvement:** Implement hierarchical multi-label classification or provide explicit secondary category routing in the UI.

### Failure Case 2: [AMBIG-03] (ambiguous)
* **Announcement Text:** *"Final year students presenting their Capstone Project Phase-1 must submit the project documentation file and clear all pending library dues before 28th November 2026."*
* **Expected Category:** `Assignments`
* **Predicted Category:** `Assignments`
* **Failure Reasons:** Missed expected anomaly flag: ['ambiguous_category']
* **Root Cause:** Vocabulary overlap between competing domains (e.g. 'Assignments' and 'Assignments'). TF-IDF weights were balanced across both sets of terms.
* **Suggested Improvement:** Implement hierarchical multi-label classification or provide explicit secondary category routing in the UI.

### Failure Case 3: [AMBIG-05] (ambiguous)
* **Announcement Text:** *"Due to Diwali holiday week, the submission of Operating Systems Assignment 4 and lab records has been postponed to 23rd November 2026."*
* **Expected Category:** `Assignments`
* **Predicted Category:** `Assignments`
* **Failure Reasons:** Missed expected anomaly flag: ['ambiguous_category']
* **Root Cause:** Vocabulary overlap between competing domains (e.g. 'Assignments' and 'Assignments'). TF-IDF weights were balanced across both sets of terms.
* **Suggested Improvement:** Implement hierarchical multi-label classification or provide explicit secondary category routing in the UI.

### Failure Case 4: [CONF-01] (conflicting_dates)
* **Announcement Text:** *"The Annual Tech Hackathon 'CodeStorm' will be held on 10th November 2026. However, team registrations on the event portal will open on 20th November 2026."*
* **Expected Category:** `Events`
* **Predicted Category:** `Events`
* **Failure Reasons:** Missed expected anomaly flag: ['conflicting_dates']
* **Root Cause:** Temporal contradiction detected between multiple dates or event sequences.
* **Suggested Improvement:** Enforce chronological validation rules using a date graph validator to highlight specific inverted date pairs.

### Failure Case 5: [CONF-03] (conflicting_dates)
* **Announcement Text:** *"The college will observe a full public holiday for Diwali on 18th November 2026. All students must physically attend the Mid-Semester Examination in Block A on 18th November 2026."*
* **Expected Category:** `Exams`
* **Predicted Category:** `Holidays`
* **Failure Reasons:** Category mismatch: expected 'Exams', got 'Holidays'
* **Root Cause:** Temporal contradiction detected between multiple dates or event sequences.
* **Suggested Improvement:** Enforce chronological validation rules using a date graph validator to highlight specific inverted date pairs.

