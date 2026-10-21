"""
EduIntent AI - Automated System Test Suite
SWYNEX Technologies AI Internship - Task 1 & Task 2
Validates:
1. Dataset structure and class balance (450 balanced queries across 9 classes)
2. Text preprocessing and normalization
3. Model artifact existence and consistency
4. Inference engine (single query, ambiguity, low confidence, code-mixing, top-3, XAI)
5. Batch prediction capability
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Ensure root src is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from preprocessing import clean_text, load_and_preprocess_data
from predict import EduIntentPredictor


class TestEduIntentSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = EduIntentPredictor(models_dir=os.path.join(PROJECT_ROOT, "models"))
        cls.dataset_path = os.path.join(PROJECT_ROOT, "dataset", "student_queries.csv")

    def test_01_dataset_integrity(self):
        """Test dataset exists, has 450 rows and exactly 9 balanced classes."""
        self.assertTrue(os.path.exists(self.dataset_path), "Dataset file not found")
        df = pd.read_csv(self.dataset_path)
        self.assertEqual(len(df), 450, f"Expected 450 rows, got {len(df)}")
        self.assertEqual(df["Category"].nunique(), 9, "Expected 9 unique categories")
        counts = df["Category"].value_counts()
        self.assertTrue((counts == 50).all(), "Classes are not balanced (expected 50 each)")

    def test_02_preprocessing(self):
        """Test text cleaning and normalization."""
        raw = "What is a Convolutional Neural Network (CNN)? Please EXPLAIN!!"
        cleaned = clean_text(raw)
        self.assertNotIn("!", cleaned)
        self.assertNotIn("?", cleaned)
        self.assertTrue(cleaned.islower())
        self.assertIn("convolutional", cleaned)

    def test_03_model_artifacts(self):
        """Test that best_model, tfidf_vectorizer, and model_metadata exist."""
        models_dir = os.path.join(PROJECT_ROOT, "models")
        self.assertTrue(os.path.exists(os.path.join(models_dir, "best_model.pkl")))
        self.assertTrue(os.path.exists(os.path.join(models_dir, "tfidf_vectorizer.pkl")))
        self.assertTrue(os.path.exists(os.path.join(models_dir, "model_metadata.json")))
        self.assertEqual(len(self.predictor.categories), 9)

    def test_04_inference_single_query(self):
        """Test inference on a standard academic query."""
        query = "Explain Dijkstra algorithm time complexity using min heap"
        res = self.predictor.predict(query)
        self.assertIn("predicted_category", res)
        self.assertEqual(res["predicted_category"], "DAA / Algorithms")
        self.assertGreater(res["confidence"], 0.40)
        self.assertIn("top_categories", res)
        self.assertEqual(len(res["top_categories"]), 3)
        self.assertIn("important_terms", res)

    def test_05_ambiguity_detection(self):
        """Test multi-subject overlap detection."""
        query = "How to train convolutional neural network in Python using PyTorch?"
        res = self.predictor.predict(query)
        # Query overlaps Deep Learning and Python / Data Science
        self.assertTrue(
            res["predicted_category"] in ["Deep Learning", "Python / Data Science", "Machine Learning"]
        )
        self.assertTrue(len(res["top_categories"]) >= 2)

    def test_06_low_confidence_guardrail(self):
        """Test low-confidence warning on vague out-of-domain input."""
        query = "hello ok thanks"
        res = self.predictor.predict(query)
        # Should flag low confidence
        self.assertTrue(
            res.get("is_low_confidence", False) or res["confidence"] < 0.45,
            "Low confidence query should be flagged"
        )

    def test_07_gujarati_code_mixing(self):
        """Test Gujarati / English code-mixing query."""
        query = "Pandas ma dataframe merge ke join kya rite karvu?"
        res = self.predictor.predict(query)
        self.assertEqual(res["predicted_category"], "Python / Data Science")
        self.assertIn("Gujarati", res.get("language", ""))

    def test_08_batch_prediction(self):
        """Test batch prediction across multiple questions."""
        queries = [
            "What is ACID properties in DBMS?",
            "How does virtual memory paging work in Operating Systems?",
            "What is Flutter Widget lifecycle?"
        ]
        results = [self.predictor.predict(q) for q in queries]
        self.assertEqual(results[0]["predicted_category"], "DBMS")
        self.assertEqual(results[1]["predicted_category"], "Operating Systems")
        self.assertEqual(results[2]["predicted_category"], "Flutter")


if __name__ == "__main__":
    unittest.main()
