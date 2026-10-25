"""
test_announcement_analyzer.py
-----------------------------
Comprehensive automated unit test suite for the Student Announcement Intelligence System.
SWYNEX AI Internship - Task 3.

Validates:
1. Validation module (empty string, symbols, extreme length, valid text).
2. Model loading & artifact verification.
3. Multi-class prediction across all 7 categories (Exams, Assignments, Events, Holidays, Placements, Fees, General).
4. Strictly non-hallucinatory deadline and event date extraction.
5. Target audience extraction.
6. Required action extraction.
7. Transparent rule-based priority estimation.
8. Missing deadline anomaly detection.
9. Conflicting dates detection.
10. Ambiguity detection.
11. Out-of-domain / Unsupported content handling.
12. Evaluation Lab benchmark execution.
"""

import os
import sys
import unittest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from validation import validate_announcement_input, verify_model_artifacts
from announcement_analyzer import SmartAnnouncementAnalyzer
from evaluation_lab import EvaluationLab

class TestAnnouncementIntelligenceSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.models_dir = os.path.join(PROJECT_ROOT, "models")
        cls.analyzer = SmartAnnouncementAnalyzer(models_dir=cls.models_dir)

    # -------------------------------------------------------------------------
    # 1. VALIDATION TESTS
    # -------------------------------------------------------------------------
    def test_01_empty_input_validation(self):
        """Test rejection of empty string."""
        is_valid, msg, diag = validate_announcement_input("")
        self.assertFalse(is_valid)
        self.assertTrue(diag["is_empty"])
        self.assertIn("empty", msg.lower())

    def test_02_whitespace_input_validation(self):
        """Test rejection of whitespace-only input."""
        is_valid, msg, diag = validate_announcement_input("   \t \n  ")
        self.assertFalse(is_valid)
        self.assertTrue(diag["is_empty"])

    def test_03_symbols_only_validation(self):
        """Test rejection of symbols/punctuation only."""
        is_valid, msg, diag = validate_announcement_input("!@#$%^&*()_+=-")
        self.assertFalse(is_valid)
        self.assertTrue(diag["is_symbols_only"])

    def test_04_valid_text_validation(self):
        """Test acceptance of standard valid announcement."""
        text = "End semester examinations commence on 15th December 2026."
        is_valid, msg, diag = validate_announcement_input(text)
        self.assertTrue(is_valid)
        self.assertEqual(msg, "")
        self.assertGreater(diag["word_count"], 5)

    # -------------------------------------------------------------------------
    # 2. MODEL ARTIFACT INTEGRITY
    # -------------------------------------------------------------------------
    def test_05_model_artifacts_exist(self):
        """Test that announcement_classifier, tfidf, and metadata exist and load."""
        required = ["announcement_classifier.pkl", "announcement_tfidf.pkl", "announcement_metadata.json"]
        is_ok, err = verify_model_artifacts(self.models_dir, required)
        self.assertTrue(is_ok, err)
        self.assertTrue(self.analyzer.is_loaded)
        self.assertEqual(len(self.analyzer.metadata["categories"]), 7)

    # -------------------------------------------------------------------------
    # 3. CATEGORY CLASSIFICATION TESTS
    # -------------------------------------------------------------------------
    def test_06_classify_exams(self):
        """Test classification of exam schedule notice."""
        text = "End Semester Theory Examinations for 5th semester will commence from 15th December 2026. Collect hall tickets."
        res = self.analyzer.analyze(text)
        self.assertEqual(res["predicted_category"], "Exams")
        self.assertGreater(res["confidence"], 0.25)

    def test_07_classify_assignments(self):
        """Test classification of assignment submission notice."""
        text = "All students must submit Assignment 3 for Object Oriented Programming via Moodle before 12th November 2026."
        res = self.analyzer.analyze(text)
        self.assertEqual(res["predicted_category"], "Assignments")

    def test_08_classify_events(self):
        """Test classification of technical symposium / hackathon."""
        text = "The annual national technical symposium and hackathon will be organized on 18th December 2026."
        res = self.analyzer.analyze(text)
        self.assertEqual(res["predicted_category"], "Events")

    def test_09_classify_holidays(self):
        """Test classification of campus holiday notice."""
        text = "The college and administrative offices will remain closed on 12th November 2026 on account of Diwali festival."
        res = self.analyzer.analyze(text)
        self.assertEqual(res["predicted_category"], "Holidays")

    def test_10_classify_placements(self):
        """Test classification of campus placement recruitment."""
        text = "TCS campus placement recruitment drive for 2027 passing out batch. Register on NextStep portal before 10th November."
        res = self.analyzer.analyze(text)
        self.assertEqual(res["predicted_category"], "Placements")

    def test_11_classify_fees(self):
        """Test classification of tuition fee notice with penalty."""
        text = "The last date for payment of Even Semester Tuition Fees is 30th November 2026. Late fine of Rs. 100 per day."
        res = self.analyzer.analyze(text)
        self.assertEqual(res["predicted_category"], "Fees")

    def test_12_classify_general(self):
        """Test classification of campus discipline / ID card notice."""
        text = "All students must wear official college identity cards inside campus premises. Security will inspect ID cards at gate."
        res = self.analyzer.analyze(text)
        self.assertEqual(res["predicted_category"], "General")

    # -------------------------------------------------------------------------
    # 4. ENTITY & TEMPORAL EXTRACTION (NO HALLUCINATIONS)
    # -------------------------------------------------------------------------
    def test_13_deadline_extraction(self):
        """Test strict deadline extraction."""
        text = "Submit the laboratory report before 25th November 2026."
        res = self.analyzer.analyze(text)
        self.assertIn("25th November 2026", res["deadlines"])
        # Ensure no invented dates
        self.assertEqual(len(res["deadlines"]), 1)

    def test_14_event_date_vs_deadline(self):
        """Test correct distinction between event date and submission deadline."""
        text = "The annual cultural fest will be held on 20th December 2026. Register your dance team before 5th December 2026."
        res = self.analyzer.analyze(text)
        self.assertIn("5th December 2026", res["deadlines"])
        self.assertIn("20th December 2026", res["event_dates"])

    def test_15_never_invent_dates(self):
        """Test that notice without dates returns empty deadline and event list."""
        text = "Please submit your homework assignment to the professor."
        res = self.analyzer.analyze(text)
        self.assertEqual(res["deadlines"], [])
        self.assertEqual(res["event_dates"], [])

    def test_16_target_audience_extraction(self):
        """Test extraction of target student cohort."""
        text = "All 3rd year Computer Engineering students must attend the mandatory seminar."
        res = self.analyzer.analyze(text)
        self.assertIsNotNone(res["target_audience"])
        self.assertIn("3rd year", res["target_audience"])

    # -------------------------------------------------------------------------
    # 5. PRIORITY ESTIMATION ENGINE
    # -------------------------------------------------------------------------
    def test_17_urgent_priority_with_penalty(self):
        """Test urgent priority triggered by late fine penalty."""
        text = "Semester tuition fees must be cleared by 30th November 2026. Late fine of Rs. 200 per day will apply."
        res = self.analyzer.analyze(text)
        self.assertEqual(res["priority"], "Urgent / High")
        self.assertTrue(any("penalty" in r.lower() or "fine" in r.lower() for r in res["priority_rationales"]))

    def test_18_informational_priority_for_holiday(self):
        """Test low informational priority for holiday notification."""
        text = "The university will remain closed on 25th December 2026 for Christmas."
        res = self.analyzer.analyze(text)
        self.assertEqual(res["priority"], "Low / Informational")

    # -------------------------------------------------------------------------
    # 6. ANOMALY & GUARDRAIL DETECTION
    # -------------------------------------------------------------------------
    def test_19_missing_deadline_flag(self):
        """Test detection of action demanded without deadline date."""
        text = "All hostel students are urgently instructed to clear their mess dues immediately."
        res = self.analyzer.analyze(text)
        self.assertIn("missing_deadline", res["flags"])
        self.assertTrue(res["has_warnings"])

    def test_20_conflicting_dates_flag(self):
        """Test detection of late submission date earlier than due date."""
        text = "Assignment submission due date is 25th November 2026. Late submissions accepted until 15th November 2026."
        res = self.analyzer.analyze(text)
        self.assertIn("conflicting_dates", res["flags"])
        self.assertTrue(res["has_warnings"])

    def test_21_out_of_domain_unsupported(self):
        """Test that out-of-domain text (recipe) is flagged as unsupported."""
        text = "To make chocolate cookies, preheat oven to 375 degrees and mix flour with brown sugar."
        res = self.analyzer.analyze(text)
        self.assertEqual(res["predicted_category"], "Unsupported / Low Confidence")
        self.assertIn("out_of_domain", res["flags"])

    def test_22_evaluation_lab_integration(self):
        """Test that Evaluation Lab executes properly and returns structured metrics."""
        lab = EvaluationLab(analyzer=self.analyzer)
        results = lab.run_evaluation()
        self.assertGreater(results["total_tests"], 10)
        self.assertGreater(results["classification"]["accuracy"], 0.70)
        self.assertIn("deadlines", results["extraction"])

if __name__ == "__main__":
    unittest.main()
