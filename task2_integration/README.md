# EduIntent AI – Model Integration (Task 2)

## Overview
EduIntent AI is an **intelligent student query classifier** that maps a free‑text academic question to one of nine subject categories and returns a confidence score, top‑3 predictions, and helpful next‑step guidance.

| Category | Icon |
|----------|------|
| Python / Data Science | 🐍 |
| Machine Learning | 🤖 |
| Deep Learning | 🧠 |
| DBMS | 🗄️ |
| Operating Systems | 💻 |
| Computer Networks | 🌐 |
| Flutter | 📱 |
| DAA / Algorithms | ⚡ |
| General Academic Query | 🎓 |

The model is a **TF‑IDF vectorizer (1‑2‑gram) + Logistic Regression** trained on 450 curated student queries (≈50 per category).

---

## Repository Structure
```
SWYNEX‑AI‑Model‑Integration/
│
├─ data/                     # CSV dataset (student_queries.csv)
├─ models/                   # Serialized artifacts
│   ├─ tfidf_vectorizer.pkl
│   ├─ classifier.pkl
│   └─ model_metadata.json
├─ results/                  # Evaluation visualizations
│   ├─ confusion_matrix.png
│   └─ model_comparison.png
├─ src/                      # Core Python modules
│   ├─ preprocessing.py       # Text cleaning & keyword extraction
│   ├─ train_model.py         # Full training & benchmarking pipeline
│   └─ predict.py             # Production inference engine
├─ examples/                 # Sample I/O markdown file
│   └─ example_inputs_outputs.md
├─ screenshots/              # UI screenshots for README / demo
│   ├─ home.png
│   ├─ prediction.png
│   └─ examples.png
├─ .streamlit/               # Streamlit config (dark theme, custom css)
│   └─ config.toml
├─ app.py                    # Streamlit UI (glass‑morphism, modern look)
├─ requirements.txt          # Python dependencies
└─ README.md                 # **This file**
```

---

## Quick Start (Local)
```bash
# 1️⃣ Clone & enter the repo
git clone https://github.com/your-username/SWYNEX-AI-Model-Integration.git
cd SWYNEX-AI-Model-Integration

# 2️⃣ Install dependencies (virtual‑env recommended)
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3️⃣ Train the model (creates artifacts under `models/` & `results/`)
python src/train_model.py

# 4️⃣ Launch the Streamlit demo
streamlit run app.py
```
Open the displayed `http://localhost:8501` in a browser.

---

## Deployment to Streamlit Community Cloud
1. **Push to GitHub** – ensure the `models/` folder (or at least the serialized `*.pkl` files) is committed.  The repository size stays well under the 1 GB limit.
2. **Create a new app** on <https://share.streamlit.io> and link it to the GitHub repo.
3. The platform automatically runs `pip install -r requirements.txt`.
4. No extra environment variables are required (the model is stored locally).
5. After a few seconds the app is live – share the generated URL.

> **Tip:** If you ever retrain the model, simply push the updated `models/` content and the hosted app will reload on the next visit.

---

## Additional Visualizations
The training script already generates two key plots:
- **Confusion Matrix** (`results/confusion_matrix.png`) – shows per‑category performance.
- **Model Comparison Chart** (`results/model_comparison.png`) – accuracy & macro‑F1 for each candidate model.

If you’d like a **Feature‑Importance** bar chart (top TF‑IDF tokens per class), run the helper script:
```bash
python src/visualize_feature_importance.py
```
It creates `results/feature_importance.png` that displays the most discriminative terms for each category.

---

## README Checklist for the Repository
- [x] Clear project description and purpose.
- [x] Architecture diagram (optional, can be added as a mermaid block).
- [x] Installation & execution steps for both local development and cloud deployment.
- [x] Dataset information (size, class distribution).
- [x] Model details (TF‑IDF + Logistic Regression, hyper‑parameters, benchmark results).
- [x] Screenshots of the UI (home, prediction, examples).
- [x] License (MIT) and citation guidance.

---

## License
MIT – feel free to fork, adapt, and use in your own projects. Credit the original author (SWYNEX AI Intern) when publishing.

---

**Enjoy building with EduIntent AI!** 🎉
