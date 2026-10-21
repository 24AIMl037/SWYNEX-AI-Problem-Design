"""
EduIntent AI - Inference & Prediction Module
SWYNEX Technologies AI Internship - Task 2: Model Integration
Author: SWYNEX AI Intern

Provides production prediction functionality with:
- Text normalization
- TF-IDF feature transformation
- Calibrated probability estimation
- Top-3 category ranking
- Low-confidence ambiguity detection (for queries like 'Explain this')
- Salient keyword extraction
"""

import os
import sys
import json
import joblib
import numpy as np

# Ensure local imports work regardless of execution location
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, CURRENT_DIR)

from preprocessing import clean_text, extract_keywords

# Category-specific learning recommendations and friendly advice
CATEGORY_GUIDES = {
    "Python / Data Science": {
        "description": "Python syntax, Pandas, NumPy, data manipulation, Matplotlib/Seaborn visualization, and scripting.",
        "icon": "🐍",
        "next_steps": "Review Pandas indexing (loc/iloc), NumPy vectorization, or Python data structures."
    },
    "Machine Learning": {
        "description": "Supervised & unsupervised learning, classification, regression, clustering, evaluation metrics, and bias-variance tradeoff.",
        "icon": "🤖",
        "next_steps": "Check train/test splitting, cross-validation, hyperparameter tuning, or Scikit-learn pipelines."
    },
    "Deep Learning": {
        "description": "Neural networks, CNNs, RNNs, Transformers, backpropagation, activation functions, and GPU training.",
        "icon": "🧠",
        "next_steps": "Explore PyTorch/TensorFlow tutorials, convolution math, loss functions, and gradient descent."
    },
    "DBMS": {
        "description": "Relational databases, SQL queries, normalization, ACID properties, indexing, transactions, and ER diagrams.",
        "icon": "🗄️",
        "next_steps": "Practice SQL joins, subqueries, group by aggregation, and B-tree indexing concepts."
    },
    "Operating Systems": {
        "description": "Processes, threads, CPU scheduling algorithms, deadlock handling, virtual memory, paging, and system calls.",
        "icon": "💻",
        "next_steps": "Review Round Robin/FCFS scheduling, Peterson's algorithm, semaphore synchronization, or page replacement."
    },
    "Computer Networks": {
        "description": "OSI 7-layer model, TCP/IP protocol suite, routing protocols, DNS, HTTP/HTTPS, subnetting, and socket programming.",
        "icon": "🌐",
        "next_steps": "Review three-way handshakes, subnet masks calculation, packet encapsulation, and port numbers."
    },
    "Flutter": {
        "description": "Cross-platform mobile development, Dart programming, widgets tree, state management (Provider/Bloc), and mobile UI.",
        "icon": "📱",
        "next_steps": "Explore StatelessWidget vs StatefulWidget, BuildContext, setState, and async Futures in Dart."
    },
    "DAA / Algorithms": {
        "description": "Design and Analysis of Algorithms, Time/Space asymptotic complexity (Big-O), dynamic programming, greedy, and graph traversals.",
        "icon": "⚡",
        "next_steps": "Analyze recurrence relations (Master Theorem), memoization patterns, BFS/DFS, and Dijkstra's algorithm."
    },
    "General Academic Query": {
        "description": "Curriculum advice, study roadmaps, exam preparation, project architecture, and general career guidance.",
        "icon": "🎓",
        "next_steps": "Consult course syllabus, academic advisor, or reference university project guidelines."
    }
}


class EduIntentPredictor:
    """Production inference engine for student academic query classification."""

    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = os.path.join(PROJECT_ROOT, "models")

        self.models_dir = models_dir
        self.tfidf_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
        self.classifier_path = os.path.join(models_dir, "classifier.pkl")
        self.metadata_path = os.path.join(models_dir, "model_metadata.json")

        self.tfidf = None
        self.classifier = None
        self.metadata = {}
        self.categories = []

        self._load_artifacts()

    def _load_artifacts(self):
        """Loads serialized model, vectorizer, and metadata."""
        if not os.path.exists(self.tfidf_path):
            raise FileNotFoundError(f"TF-IDF Vectorizer not found at {self.tfidf_path}. Please run train_model.py first.")
        if not os.path.exists(self.classifier_path):
            raise FileNotFoundError(f"Classifier not found at {self.classifier_path}. Please run train_model.py first.")

        self.tfidf = joblib.load(self.tfidf_path)
        self.classifier = joblib.load(self.classifier_path)

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            self.categories = self.metadata.get("categories", list(self.classifier.classes_))
        else:
            self.categories = list(self.classifier.classes_)

    def predict(self, query: str, confidence_threshold: float = 0.35) -> dict:
        """
        Predicts category, confidence, top 3 ranking, and keywords for a query.

        Parameters:
        -----------
        query: str
            Raw question string entered by student.
        confidence_threshold: float
            Minimum probability below which a query is marked as low confidence (ambiguous).

        Returns:
        --------
        dict containing classification results, confidence, top 3 list, and guidance.
        """
        raw_query = query if isinstance(query, str) else ""
        cleaned = clean_text(raw_query)
        words = cleaned.split()

        # Handle empty or ultra-short input
        if len(words) == 0:
            return {
                "input_query": raw_query,
                "cleaned_query": cleaned,
                "predicted_category": "Unknown",
                "confidence": 0.0,
                "is_low_confidence": True,
                "warning_message": "⚠ No query provided. Please enter a valid academic question.",
                "top_predictions": [],
                "keywords": [],
                "category_meta": {}
            }

        # Check for vague / overly brief queries (e.g. "Explain this", "help me")
        is_vague_input = len(words) <= 2 and not any(
            w in {"cnn", "rnn", "sql", "dbms", "os", "tcp", "udp", "dart", "bfs", "dfs", "relu", "svm"}
            for w in words
        )

        # TF-IDF Feature Extraction
        query_vec = self.tfidf.transform([cleaned])

        # Get calibrated probability distribution
        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(query_vec)[0]
        else:
            # Fallback for models without direct predict_proba (decision function softmax)
            scores = self.classifier.decision_function(query_vec)[0]
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / np.sum(exp_scores)

        # Sort classes descending by probability
        classes = self.classifier.classes_
        sorted_indices = np.argsort(probs)[::-1]

        top_predictions = []
        for idx in sorted_indices[:3]:
            cat_name = str(classes[idx])
            prob_val = float(probs[idx])
            top_predictions.append({
                "category": cat_name,
                "probability": round(prob_val, 4),
                "confidence_percent": round(prob_val * 100, 2),
                "icon": CATEGORY_GUIDES.get(cat_name, {}).get("icon", "📚")
            })

        primary_cat = top_predictions[0]["category"]
        primary_conf = top_predictions[0]["probability"]
        primary_conf_pct = top_predictions[0]["confidence_percent"]

        # Ambiguity / Low-confidence check
        is_low_confidence = False
        warning_message = None

        if is_vague_input or primary_conf < confidence_threshold:
            is_low_confidence = True
            warning_message = (
                "⚠ Low Confidence: The query is not specific enough or covers multiple subjects. "
                "Please add more technical context, error messages, or specify the topic."
            )

        # Extract salient keywords from query
        keywords = extract_keywords(cleaned, vectorizer=self.tfidf, top_n=5)

        return {
            "input_query": raw_query,
            "cleaned_query": cleaned,
            "predicted_category": primary_cat,
            "confidence": primary_conf,
            "confidence_percent": primary_conf_pct,
            "is_low_confidence": is_low_confidence,
            "warning_message": warning_message,
            "top_predictions": top_predictions,
            "keywords": keywords,
            "category_meta": CATEGORY_GUIDES.get(primary_cat, {
                "description": "Academic coursework subject.",
                "icon": "🎓",
                "next_steps": "Refer to standard academic textbooks and lecture notes."
            })
        }


# Standalone helper function
def predict_query(query: str, confidence_threshold: float = 0.35) -> dict:
    predictor = EduIntentPredictor()
    return predictor.predict(query, confidence_threshold=confidence_threshold)


if __name__ == "__main__":
    test_queries = [
        "What is the difference between supervised and unsupervised learning?",
        "How do I fix SettingWithCopyWarning in Pandas DataFrame?",
        "What is a primary key and foreign key in SQL DBMS?",
        "Explain backpropagation algorithm in CNN deep learning.",
        "What is process scheduling and deadlock in Operating Systems?",
        "Explain this",  # Low-confidence test case
        "How can I implement CNN using Python?"  # Multi-disciplinary test case
    ]

    print("=" * 65)
    print(" EduIntent AI - Inference Smoke Test")
    print("=" * 65)

    predictor = EduIntentPredictor()
    for q in test_queries:
        res = predictor.predict(q)
        print(f"\nQuery:      \"{res['input_query']}\"")
        print(f"Predicted:  {res['predicted_category']} (Confidence: {res['confidence_percent']}%)")
        print(f"Keywords:   {', '.join(res['keywords'])}")
        if res['is_low_confidence']:
            print(f"Warning:    {res['warning_message']}")
        print(f"Top 3:      " + " | ".join([f"{p['category']}: {p['confidence_percent']}%" for p in res['top_predictions']]))
