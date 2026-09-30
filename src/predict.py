import os
import sys
import json
import re
import joblib
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from preprocessing import clean_text

GUJARATI_ROMAN_KEYWORDS = {"su", "che", "chhe", "kem", "mate", "aave", "karo", "ma", "ketyare", "syu", "suche"}
HINGLISH_KEYWORDS = {"kya", "hai", "kaise", "hota", "batao", "samjhao", "karna", "sir", "bhai"}

class EduIntentPredictor:
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, "models")
        
        vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
        model_path = os.path.join(models_dir, "best_model.pkl")
        meta_path = os.path.join(models_dir, "model_metadata.json")

        if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
            raise FileNotFoundError("Model artifacts not found. Run 'train.py' first.")

        self.vectorizer = joblib.load(vectorizer_path)
        self.model = joblib.load(model_path)
        
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)
            
        self.categories = self.metadata["categories"]
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())

    def _detect_language(self, query: str) -> str:
        """
        Detects query language: English, Gujarati / Code-Mixed, or Hinglish.
        """
        # Check for Gujarati Unicode script range (U+0A80 to U+0AFF)
        if re.search(r'[\u0A80-\u0AFF]', query):
            return "Gujarati (Native Script)"
        
        tokens = set(re.findall(r'\b\w+\b', query.lower()))
        if tokens.intersection(GUJARATI_ROMAN_KEYWORDS):
            return "Gujarati / English Code-Mixed"
        elif tokens.intersection(HINGLISH_KEYWORDS):
            return "Hinglish / Code-Mixed"
        else:
            return "English"

    def _extract_important_terms(self, cleaned_query: str, top_category: str):
        """
        Extracts key n-gram terms in the query that contributed most to the prediction.
        """
        # Transform text to sparse vector
        vec = self.vectorizer.transform([cleaned_query])
        feature_indices = vec.nonzero()[1]
        
        if len(feature_indices) == 0:
            return []

        # Extract non-zero TF-IDF terms in user query
        tfidf_scores = vec.toarray()[0]
        query_terms = [(self.feature_names[idx], tfidf_scores[idx]) for idx in feature_indices]
        
        # Sort terms by TF-IDF weight
        sorted_terms = sorted(query_terms, key=lambda x: x[1], reverse=True)
        return [term for term, score in sorted_terms[:5]]

    def predict(self, query: str):
        """
        Comprehensive prediction with Explainability, Ambiguity Detection,
        Low-Confidence Warning, and Language Detection.
        """
        cleaned = clean_text(query)
        lang_detected = self._detect_language(query)

        if not cleaned:
            return {
                "query": query,
                "cleaned_query": "",
                "predicted_category": "Unknown / Low Confidence",
                "confidence": 0.0,
                "is_low_confidence": True,
                "is_ambiguous": False,
                "language": lang_detected,
                "important_terms": [],
                "top_categories": [],
                "user_message": "⚠️ Empty query provided. Please type a valid academic question."
            }

        vec = self.vectorizer.transform([cleaned])
        
        # Check if vectorizer produced zero non-zero entries (all OOV words)
        if vec.nnz == 0:
            return {
                "query": query,
                "cleaned_query": cleaned,
                "predicted_category": "General Academic Query",
                "confidence": 0.30,
                "is_low_confidence": True,
                "is_ambiguous": False,
                "language": lang_detected,
                "important_terms": [],
                "top_categories": [{"category": "General Academic Query", "probability": 0.30}],
                "user_message": "⚠️ Low Confidence Prediction (Out-of-Vocabulary Terms). Please add more subject-specific keywords."
            }

        # Predict category
        pred_label = self.model.predict(vec)[0]

        # Get probabilities across all classes
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vec)[0]
            classes = self.model.classes_
            prob_dict = {cls: float(prob) for cls, prob in zip(classes, probs)}
            confidence = float(prob_dict[pred_label])
            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            top_categories = [{"category": cat, "probability": round(prob, 4)} for cat, prob in sorted_probs[:3]]
        else:
            confidence = 1.0
            sorted_probs = [(pred_label, 1.0)]
            top_categories = [{"category": pred_label, "probability": 1.0}]

        # Extract Explainable AI Important Terms
        important_terms = self._extract_important_terms(cleaned, pred_label)

        # Ambiguity Detection (Top 1 vs Top 2 probability gap < 0.25)
        is_ambiguous = False
        related_category = None
        if len(sorted_probs) >= 2:
            p1 = sorted_probs[0][1]
            p2 = sorted_probs[1][1]
            if (p1 - p2) < 0.25 and p1 < 0.75:
                is_ambiguous = True
                related_category = sorted_probs[1][0]

        # Low Confidence Warning Threshold (< 45% or short vague query without domain keywords)
        word_count = len(cleaned.split())
        is_low_confidence = False
        if confidence < 0.45 or (word_count < 3 and len(important_terms) == 0):
            is_low_confidence = True

        # Formulate intuitive user message
        if is_low_confidence:
            user_message = "⚠️ Low Confidence Prediction (< 45%). The query is brief or vague. Please add subject details."
        elif is_ambiguous:
            user_message = f"🔀 Ambiguous Multi-Subject Query Detected! Primary: '{pred_label}' | Related: '{related_category}'."
        else:
            user_message = f"✅ High Confidence Match ({confidence*100:.1f}%)."

        return {
            "query": query,
            "cleaned_query": cleaned,
            "predicted_category": pred_label,
            "confidence": round(confidence, 4),
            "is_low_confidence": is_low_confidence,
            "is_ambiguous": is_ambiguous,
            "related_category": related_category,
            "language": lang_detected,
            "important_terms": important_terms,
            "top_categories": top_categories,
            "user_message": user_message
        }

if __name__ == "__main__":
    predictor = EduIntentPredictor()
    test_queries = [
        "Why am I getting a KeyError while accessing a Pandas column?",
        "How can I implement CNN using Python?",
        "Can you explain this?",
        "Pandas ma KeyError aave chhe su karvu?",
        "What is deadlock in OS?"
    ]

    print("\n==========================================")
    print("   EduIntent AI Unique Features Inference Test   ")
    print("==========================================\n")
    for q in test_queries:
        res = predictor.predict(q)
        print(f"Query: '{res['query']}'")
        print(f"Language: {res['language']}")
        print(f"Predicted Category: {res['predicted_category']} (Confidence: {res['confidence']*100:.1f}%)")
        print(f"Important Terms: {res['important_terms']}")
        print(f"Status Message: {res['user_message']}")
        print("-" * 65)
