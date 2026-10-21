"""
Streamlit UI for EduIntent AI – Student Query Classification
"""

import json
import os
import streamlit as st
from src.predict import EduIntentPredictor

# ---------------------------------------------------------------------------
# Page configuration – modern look
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EduIntent AI – Query Classifier",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS – glassmorphism + dark theme accents
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #1e1e2f, #2a2a3b);
        color: #e0e0ff;
    }
    .stButton>button {
        background: rgba(255,255,255,0.1);
        color: #fff;
        border-radius: 8px;
        border: 1px solid #4e4e9a;
        font-weight: 600;
    }
    .stButton>button:hover {
        background: rgba(255,255,255,0.2);
    }
    .stTextInput>div>div>input {
        background: rgba(255,255,255,0.05);
        color: #e0e0ff;
        border: 1px solid #4e4e9a;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load model – singleton for efficiency
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_predictor() -> EduIntentPredictor:
    # The model files live under the same "models" folder as this script
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
    return EduIntentPredictor(models_dir=models_dir)

predictor = get_predictor()

# ---------------------------------------------------------------------------
# Sidebar – Model & Dataset Overview
# ---------------------------------------------------------------------------
st.sidebar.title("📊 Model Information")
metadata_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "model_metadata.json"))
if os.path.exists(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    st.sidebar.markdown(f"**Model**: {meta.get('primary_model', 'Logistic Regression')}")
    st.sidebar.markdown(f"**Features**: {meta.get('feature_extraction', 'TF‑IDF (1‑2‑gram)')}")
    st.sidebar.markdown(f"**Categories**: {meta.get('num_categories', 9)}")
    st.sidebar.markdown(f"**Test Accuracy**: {meta.get('test_accuracy', 0)*100:.2f}%")
    st.sidebar.markdown(f"**Weighted F1**: {meta.get('weighted_f1', 0)*100:.2f}%")
else:
    st.sidebar.info("Model metadata not found – ensure the training pipeline has run.")

# ---------------------------------------------------------------------------
# Main UI – Input & Prediction
# ---------------------------------------------------------------------------
st.title("EduIntent AI – Intelligent Student Query Classifier")
st.subheader("Enter an academic question and discover its subject area instantly.")

# Example queries – quick selection
example_queries = [
    "What is overfitting in machine learning?",
    "Explain primary key and foreign key in SQL.",
    "How does backpropagation work in a neural network?",
    "What is Round Robin scheduling in Operating Systems?",
    "Explain the difference between supervised and unsupervised learning.",
]
selected_example = st.selectbox("🔎 Try an example query", ["-- select --"] + example_queries)

if selected_example != "-- select --":
    query = selected_example
else:
    query = st.text_area("🖊️ Student Question", height=150, placeholder="e.g., What is a foreign key?")

if st.button("🚀 Predict Category"):
    if not query.strip():
        st.warning("⚠ Please type a question before predicting.")
    else:
        with st.spinner("Classifying…"):
            result = predictor.predict(query)
        # Primary result card
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"## {result['predicted_category']} {result['category_meta'].get('icon', '')}")
            st.markdown(f"**Confidence:** {result['confidence_percent']:.2f}%")
            st.markdown(f"**Keywords:** {', '.join(result['keywords'])}")
        with col2:
            st.metric(label="Confidence", value=f"{result['confidence_percent']:.2f}%")
        # Low‑confidence warning
        if result.get('is_low_confidence'):
            st.error(result.get('warning_message', "Low confidence – please provide more details."))
        # Top‑3 table
        st.subheader("📊 Top‑3 Predictions")
        top_df = {
            "Category": [p['category'] for p in result['top_predictions']],
            "Confidence": [f"{p['confidence_percent']:.2f}%" for p in result['top_predictions']],
            "Icon": [p.get('icon', '') for p in result['top_predictions']],
        }
        st.table(top_df)
        # Guidance section
        guide = result['category_meta']
        if guide:
            st.subheader("🧭 Next Steps")
            st.markdown(f"{guide.get('description', '')}")
            st.markdown(f"**Suggested focus:** {guide.get('next_steps', '')}")

st.caption("*Model trained on 450 curated student queries across 9 academic domains. Powered by TF‑IDF + Logistic Regression.*")
