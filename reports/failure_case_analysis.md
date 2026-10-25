# Student Announcement Intelligence System — Failure-Case Analysis

Total failure cases recorded: **5** out of 30 tests.

This document provides a systematic post-mortem analysis of failure modes encountered during rigorous evaluation.

---

## Failure Case #1: AMBIG-01 (ambiguous)

- **Input Announcement:** `Students with backlogs must pay the supplementary examination fee of Rs. 500 and submit the re-exam registration form before 18th November 2026.`
- **Expected Category:** `Exams`
- **Predicted Category:** `Fees`
- **Symptoms & Discrepancies:** Missed expected anomaly flag: ['ambiguous_category']
- **Root Cause Analysis:** Vocabulary overlap between competing domains (e.g. 'Exams' and 'Fees'). TF-IDF weights were balanced across both sets of terms.
- **Proposed Engineering Solution:** Implement hierarchical multi-label classification or provide explicit secondary category routing in the UI.

---

## Failure Case #2: AMBIG-03 (ambiguous)

- **Input Announcement:** `Final year students presenting their Capstone Project Phase-1 must submit the project documentation file and clear all pending library dues before 28th November 2026.`
- **Expected Category:** `Assignments`
- **Predicted Category:** `Assignments`
- **Symptoms & Discrepancies:** Missed expected anomaly flag: ['ambiguous_category']
- **Root Cause Analysis:** Vocabulary overlap between competing domains (e.g. 'Assignments' and 'Assignments'). TF-IDF weights were balanced across both sets of terms.
- **Proposed Engineering Solution:** Implement hierarchical multi-label classification or provide explicit secondary category routing in the UI.

---

## Failure Case #3: AMBIG-05 (ambiguous)

- **Input Announcement:** `Due to Diwali holiday week, the submission of Operating Systems Assignment 4 and lab records has been postponed to 23rd November 2026.`
- **Expected Category:** `Assignments`
- **Predicted Category:** `Assignments`
- **Symptoms & Discrepancies:** Missed expected anomaly flag: ['ambiguous_category']
- **Root Cause Analysis:** Vocabulary overlap between competing domains (e.g. 'Assignments' and 'Assignments'). TF-IDF weights were balanced across both sets of terms.
- **Proposed Engineering Solution:** Implement hierarchical multi-label classification or provide explicit secondary category routing in the UI.

---

## Failure Case #4: CONF-01 (conflicting_dates)

- **Input Announcement:** `The Annual Tech Hackathon 'CodeStorm' will be held on 10th November 2026. However, team registrations on the event portal will open on 20th November 2026.`
- **Expected Category:** `Events`
- **Predicted Category:** `Events`
- **Symptoms & Discrepancies:** Missed expected anomaly flag: ['conflicting_dates']
- **Root Cause Analysis:** Temporal contradiction detected between multiple dates or event sequences.
- **Proposed Engineering Solution:** Enforce chronological validation rules using a date graph validator to highlight specific inverted date pairs.

---

## Failure Case #5: CONF-03 (conflicting_dates)

- **Input Announcement:** `The college will observe a full public holiday for Diwali on 18th November 2026. All students must physically attend the Mid-Semester Examination in Block A on 18th November 2026.`
- **Expected Category:** `Exams`
- **Predicted Category:** `Holidays`
- **Symptoms & Discrepancies:** Category mismatch: expected 'Exams', got 'Holidays'
- **Root Cause Analysis:** Temporal contradiction detected between multiple dates or event sequences.
- **Proposed Engineering Solution:** Enforce chronological validation rules using a date graph validator to highlight specific inverted date pairs.

---

