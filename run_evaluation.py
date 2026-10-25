"""
run_evaluation.py
-----------------
Runs the Evaluation Lab benchmark and saves reports to reports/ and results/.
"""

import os
import sys
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CURRENT_DIR, "src"))

from evaluation_lab import EvaluationLab

def main():
    reports_dir = os.path.join(CURRENT_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    print("Initializing Evaluation Lab...")
    lab = EvaluationLab()
    print("Running comprehensive evaluation on test suite...")
    results = lab.run_evaluation()

    report_md = lab.generate_markdown_report(results)
    
    # Save markdown report
    report_path = os.path.join(reports_dir, "evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Saved evaluation report to: {report_path}")

    # Save JSON raw results
    json_path = os.path.join(reports_dir, "evaluation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved raw evaluation results to: {json_path}")

    # Save failure-case documentation
    failure_path = os.path.join(reports_dir, "failure_case_analysis.md")
    failures = results.get("failure_cases", [])
    with open(failure_path, "w", encoding="utf-8") as f:
        f.write("# Student Announcement Intelligence System — Failure-Case Analysis\n\n")
        f.write(f"Total failure cases recorded: **{len(failures)}** out of {results['total_tests']} tests.\n\n")
        f.write("This document provides a systematic post-mortem analysis of failure modes encountered during rigorous evaluation.\n\n---\n\n")
        for idx, item in enumerate(failures, 1):
            f.write(f"## Failure Case #{idx}: {item['id']} ({item['test_type']})\n\n")
            f.write(f"- **Input Announcement:** `{item['text']}`\n")
            f.write(f"- **Expected Category:** `{item['expected_category']}`\n")
            f.write(f"- **Predicted Category:** `{item['predicted_category']}`\n")
            f.write(f"- **Symptoms & Discrepancies:** {', '.join(item['failure_reasons'])}\n")
            f.write(f"- **Root Cause Analysis:** {item['root_cause']}\n")
            f.write(f"- **Proposed Engineering Solution:** {item['suggested_improvement']}\n\n---\n\n")
    print(f"Saved failure case documentation to: {failure_path}")

    print("\nSummary of results:")
    print(f"Accuracy: {results['classification']['accuracy']*100:.2f}%")
    print(f"Macro F1: {results['classification']['f1_macro']:.4f}")
    print(f"Deadline F1: {results['extraction']['deadlines']['f1']*100:.1f}%")
    print(f"Event Date F1: {results['extraction']['event_dates']['f1']*100:.1f}%")
    print(f"Guardrail Detection Rate: {results['extraction']['guardrail_detection_rate']*100:.1f}%")

if __name__ == "__main__":
    main()
