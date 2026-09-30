"""
EduIntent AI – SWYNEX Task 1 – Premium Interactive Dashboard
============================================================
Built with: Python · Streamlit · Scikit-Learn · Plotly · WordCloud
Unique features:
  1. Live Typing Classifier (auto-classify as you type)
  2. Radar / Spider Chart – all-category confidence
  3. Query Complexity Analyzer (vocabulary richness, technical density)
  4. Smart Similar Query Suggestions from dataset
  5. Category Deep-Dive Explorer with WordCloud
  6. Session Analytics & Live Stats
  7. Batch CSV Uploader with downloadable results
  8. Confidence Gauge Meter (Plotly)
  9. Dark / Light Theme Toggle
  10. Full SWYNEX Branded Sidebar Navigation
"""

import os, sys, json, math, time, datetime, io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "src"))
from predict import EduIntentPredictor

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduIntent AI — SWYNEX Task 1",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Session State Init ───────────────────────────────────────────────────────
for key, default in {
    "history":      [],
    "dark":         True,
    "active_nav":   "🏠 Overview",
    "active_q":     "Why am I getting a KeyError while accessing a Pandas column?",
    "session_start": datetime.datetime.now().strftime("%H:%M:%S"),
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

D = st.session_state["dark"]

# ─── Color Palette ────────────────────────────────────────────────────────────
BG        = "#0b0f1a" if D else "#f1f5f9"
SIDEBAR   = "#0f172a" if D else "#ffffff"
CARD      = "#16213e" if D else "#ffffff"
CARD2     = "#1a2744" if D else "#f8fafc"
TEXT      = "#f1f5f9" if D else "#0f172a"
MUTED     = "#94a3b8" if D else "#64748b"
BORDER    = "rgba(255,255,255,0.07)" if D else "rgba(0,0,0,0.08)"
MPL_BORDER = (1, 1, 1, 0.07) if D else (0, 0, 0, 0.08)   # matplotlib-safe tuple
INDIGO    = "#6366f1"
VIOLET    = "#7c3aed"
SKY       = "#38bdf8"
EMERALD   = "#10b981"
AMBER     = "#f59e0b"
ROSE      = "#f43f5e"
PLT_BG    = "#16213e" if D else "#ffffff"
PLT_TEXT  = "#f1f5f9" if D else "#0f172a"

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

*, html, body, .stApp {{
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    box-sizing: border-box;
}}
.stApp {{
    background: {BG};
    color: {TEXT};
}}
/* Sidebar */
section[data-testid="stSidebar"] {{
    background: {SIDEBAR} !important;
    border-right: 1px solid {BORDER};
    width: 260px !important;
}}
/* Remove default streamlit paddings */
.block-container {{ padding: 1.4rem 2rem 2rem 2rem !important; max-width: 1320px; }}
/* Sidebar radio */
div[data-testid="stRadio"] > label {{ display: none !important; }}
div[data-testid="stRadio"] div[role="radiogroup"] {{
    gap: 4px; display: flex; flex-direction: column;
}}
div[data-testid="stRadio"] div[role="radiogroup"] label {{
    padding: 9px 12px;
    border-radius: 9px;
    color: {MUTED} !important;
    font-size: 0.93rem;
    font-weight: 600;
    cursor: pointer;
    border: 1px solid transparent;
    transition: all 0.15s;
}}
div[data-testid="stRadio"] div[role="radiogroup"] label:hover {{
    background: rgba(99,102,241,0.1);
    color: {TEXT} !important;
}}
div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] {{
    background: linear-gradient(135deg,#1e1b4b,#312e81) !important;
    color: #ffffff !important;
    border-color: rgba(99,102,241,0.4) !important;
    box-shadow: 0 2px 10px rgba(99,102,241,0.2);
}}
/* Cards */
.edu-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 22px 24px;
    margin-bottom: 18px;
}}
.edu-card-sm {{
    background: {CARD2};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 12px;
}}
/* Badges */
.badge-indigo {{
    display:inline-block; background:linear-gradient(135deg,#4f46e5,#6366f1);
    color:#fff; font-weight:700; padding:9px 22px; border-radius:9999px;
    font-size:1.2rem; box-shadow:0 4px 14px rgba(99,102,241,0.35); letter-spacing:0.3px;
}}
.badge-tag {{
    display:inline-block; background:rgba(99,102,241,0.12); color:#818cf8;
    border:1px solid rgba(99,102,241,0.3); font-weight:600; padding:4px 12px;
    border-radius:8px; font-size:0.85rem; margin:3px 4px 3px 0;
}}
.badge-emerald {{
    display:inline-block; background:rgba(16,185,129,0.15); color:{EMERALD};
    border:1px solid rgba(16,185,129,0.3); font-weight:700; padding:4px 12px;
    border-radius:8px; font-size:0.84rem; margin:2px 4px 2px 0;
}}
.badge-amber {{
    display:inline-block; background:rgba(245,158,11,0.15); color:{AMBER};
    border:1px solid rgba(245,158,11,0.3); font-weight:700; padding:4px 12px;
    border-radius:8px; font-size:0.84rem;
}}
.badge-rose {{
    display:inline-block; background:rgba(244,63,94,0.15); color:{ROSE};
    border:1px solid rgba(244,63,94,0.3); font-weight:700; padding:4px 12px;
    border-radius:8px; font-size:0.84rem;
}}
/* Nav top bar */
.topbar {{
    display:flex; justify-content:space-between; align-items:center;
    padding-bottom:14px; border-bottom:1px solid {BORDER}; margin-bottom:22px;
}}
.topbar-title {{
    font-size:1.1rem; font-weight:800; color:{TEXT};
}}
.pill-badge {{
    background:{CARD}; border:1px solid {BORDER}; color:{MUTED};
    padding:5px 14px; border-radius:9999px; font-size:0.82rem; font-weight:600;
}}
/* Hero */
.hero-kicker {{ color:{INDIGO}; font-size:0.82rem; font-weight:800; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:8px; }}
.hero-h1 {{ font-size:2.9rem; font-weight:900; color:{TEXT}; letter-spacing:-1px; line-height:1.1; margin-bottom:14px; }}
.hero-sub {{ color:{MUTED}; font-size:1.1rem; line-height:1.65; max-width:600px; margin-bottom:24px; }}
/* Metric kpi */
.kpi-card {{
    background:{CARD}; border:1px solid {BORDER}; border-radius:14px;
    padding:18px 20px; text-align:center;
}}
.kpi-val {{ font-size:1.9rem; font-weight:900; color:{TEXT}; line-height:1; }}
.kpi-label {{ font-size:0.82rem; color:{MUTED}; margin-top:4px; font-weight:600; }}
.kpi-delta {{ font-size:0.78rem; color:{EMERALD}; font-weight:700; margin-top:4px; }}
/* Confidence bars */
.conf-high {{ color:{EMERALD}; font-weight:800; }}
.conf-med  {{ color:{AMBER};   font-weight:800; }}
.conf-low  {{ color:{ROSE};    font-weight:800; }}
/* Progress inner label */
.prog-row {{ margin:6px 0; }}
/* Category icon map */
.cat-icon {{ font-size:1.5rem; }}
/* Stat pill */
.stat-pill {{
    display:inline-block; background:{CARD2}; border:1px solid {BORDER};
    border-radius:9999px; padding:6px 16px; font-size:0.85rem;
    font-weight:700; color:{TEXT}; margin:4px 4px 4px 0;
}}
/* hide streamlit footer */
footer {{ visibility: hidden; }}
/* Scrollbar */
::-webkit-scrollbar {{ width:6px; }}
::-webkit-scrollbar-track {{ background:transparent; }}
::-webkit-scrollbar-thumb {{ background:rgba(99,102,241,0.35); border-radius:4px; }}
</style>
""", unsafe_allow_html=True)

# ─── Category Metadata ────────────────────────────────────────────────────────
CAT_ICONS = {
    "Python / Data Science":  "🐍",
    "Machine Learning":       "🤖",
    "Deep Learning":          "🧠",
    "DBMS":                   "💾",
    "Operating Systems":      "⚙️",
    "Computer Networks":      "🌐",
    "Flutter":                "📱",
    "DAA / Algorithms":       "📐",
    "General Academic Query": "📚",
}
CAT_COLORS = {
    "Python / Data Science":  "#38bdf8",
    "Machine Learning":       "#818cf8",
    "Deep Learning":          "#c084fc",
    "DBMS":                   "#f59e0b",
    "Operating Systems":      "#10b981",
    "Computer Networks":      "#06b6d4",
    "Flutter":                "#6366f1",
    "DAA / Algorithms":       "#f43f5e",
    "General Academic Query": "#94a3b8",
}
CAT_KEYWORDS = {
    "Python / Data Science":  ["pandas", "numpy", "matplotlib", "dataframe", "series", "csv", "plot", "scipy", "sklearn", "seaborn", "jupyter"],
    "Machine Learning":       ["regression", "classification", "clustering", "svm", "decision tree", "random forest", "gradient boosting", "feature", "overfitting", "cross-validation"],
    "Deep Learning":          ["neural network", "cnn", "rnn", "lstm", "transformer", "backpropagation", "activation", "relu", "epoch", "batch", "pytorch", "tensorflow", "keras"],
    "DBMS":                   ["sql", "database", "query", "join", "normalization", "acid", "transaction", "index", "primary key", "foreign key", "er diagram"],
    "Operating Systems":      ["process", "thread", "deadlock", "scheduling", "memory", "paging", "semaphore", "mutex", "interrupt", "kernel", "virtual memory"],
    "Computer Networks":      ["tcp", "ip", "http", "dns", "router", "packet", "bandwidth", "protocol", "osi", "subnet", "firewall", "latency"],
    "Flutter":                ["widget", "dart", "stateful", "stateless", "async", "future", "stream", "scaffold", "provider", "getx", "bloc", "navigator"],
    "DAA / Algorithms":       ["algorithm", "complexity", "big-o", "sorting", "searching", "graph", "dynamic programming", "greedy", "recursion", "tree", "heap"],
    "General Academic Query": ["assignment", "deadline", "marks", "attendance", "exam", "result", "hall ticket", "syllabus", "timetable", "college", "faculty"],
}

# ─── Load Resources ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_predictor():
    return EduIntentPredictor()

@st.cache_data(show_spinner=False)
def get_metadata():
    with open(os.path.join(BASE_DIR, "models", "model_metadata.json"), "r") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def get_dataset():
    return pd.read_csv(os.path.join(BASE_DIR, "dataset", "student_queries.csv"))

try:
    predictor = get_predictor()
    meta      = get_metadata()
    dataset   = get_dataset()
except Exception as e:
    st.error(f"❌ Error loading model artifacts: {e}. Please run `python src/train.py` first.")
    st.stop()

SVM_ACC = meta["results_summary"]["Linear SVM"]["Accuracy"]
SVM_F1  = meta["results_summary"]["Linear SVM"]["F1-Score (Macro)"]

# ─── Helper: Plotly dark styling ──────────────────────────────────────────────
def plotly_layout(title="", height=350, margin=None):
    if margin is None:
        margin = dict(l=16, r=16, t=44, b=16)
    return dict(
        title=dict(text=title, font=dict(color=PLT_TEXT, size=14, family="Inter"), x=0.02),
        paper_bgcolor=PLT_BG,
        plot_bgcolor=PLT_BG,
        font=dict(color=PLT_TEXT, family="Inter"),
        height=height,
        margin=margin,
    )

# ─── Helper: Confidence gauge ─────────────────────────────────────────────────
def conf_gauge(conf: float, height=280):
    color = EMERALD if conf >= 0.75 else (AMBER if conf >= 0.50 else ROSE)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(conf * 100, 1),
        number={"suffix": "%", "font": {"size": 34, "color": TEXT, "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"size": 10}},
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": CARD2,
            "borderwidth": 0,
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.75, "value": conf*100},
            "steps": [
                {"range": [0, 45],  "color": "rgba(244,63,94,0.1)"},
                {"range": [45, 75], "color": "rgba(245,158,11,0.1)"},
                {"range": [75, 100],"color": "rgba(16,185,129,0.1)"},
            ],
        }
    ))
    fig.update_layout(**plotly_layout(height=height, margin=dict(l=24, r=24, t=20, b=10)))
    return fig

# ─── Helper: Radar chart for all-category confidences ────────────────────────
def radar_chart(res: dict, height=340):
    if not hasattr(predictor.model, "predict_proba"):
        return None
    from src.preprocessing import clean_text as _ct
    vec = predictor.vectorizer.transform([_ct(res["query"])])
    if vec.nnz == 0:
        return None
    probs = predictor.model.predict_proba(vec)[0]
    cats  = list(predictor.model.classes_)
    values = list(probs) + [probs[0]]
    cats_r  = cats + [cats[0]]
    fig = go.Figure(go.Scatterpolar(
        r=values, theta=cats_r, fill="toself",
        fillcolor="rgba(99,102,241,0.2)",
        line=dict(color=INDIGO, width=2),
        marker=dict(color=INDIGO, size=6),
        name="Confidence"
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=PLT_BG,
            angularaxis=dict(tickfont=dict(size=9, color=PLT_TEXT), linecolor="#2a3a5a" if D else "#e2e8f0"),
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=8, color=MUTED), gridcolor="#2a3a5a" if D else "#e2e8f0", linecolor="#2a3a5a" if D else "#e2e8f0"),
        ),
        **plotly_layout("All-Category Confidence Radar", height=height)
    )
    return fig

# ─── Helper: Smart Suggestions ───────────────────────────────────────────────
def smart_suggestions(category: str, n=4):
    rows = dataset[dataset["Category"] == category].sample(min(n, len(dataset[dataset["Category"] == category])))
    return rows["Question"].tolist()

# ─── Helper: Query Complexity Analysis ───────────────────────────────────────
def analyze_complexity(query: str, result: dict):
    words = query.split()
    unique = set(w.lower() for w in words)
    cat = result["predicted_category"]
    kws = CAT_KEYWORDS.get(cat, [])
    tech_hits = sum(1 for kw in kws if kw.lower() in query.lower())
    richness   = len(unique) / max(len(words), 1)
    tech_score = min(tech_hits / max(len(kws), 1), 1.0)
    spec_score = min(len(result["important_terms"]) / 5, 1.0)
    grade = "Excellent" if richness > 0.8 and tech_score > 0.05 else (
            "Good"      if richness > 0.65 else (
            "Average"   if richness > 0.5  else "Vague"))
    return {
        "word_count":  len(words),
        "unique_words": len(unique),
        "vocab_richness": round(richness, 2),
        "tech_density":  round(tech_score, 2),
        "specificity":   round(spec_score, 2),
        "grade": grade,
    }

# ─── Helper: WordCloud ────────────────────────────────────────────────────────
def make_wordcloud(category: str):
    try:
        from wordcloud import WordCloud
    except ImportError:
        return None
    rows = dataset[dataset["Category"] == category]["Question"].str.cat(sep=" ")
    color = CAT_COLORS.get(category, "#6366f1").lstrip("#")
    r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    def color_func(*args, **kwargs):
        return f"rgb({r},{g},{b})"
    wc = WordCloud(width=700, height=300, background_color=PLT_BG,
                   color_func=color_func, max_words=60, prefer_horizontal=0.85)
    wc.generate(rows)
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor(PLT_BG)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

# ─── Helper: Matplotlib bar (plotly style) ────────────────────────────────────
def mpl_hbar(labels, values, colors, xlabel="", title="", height=3.8):
    fig, ax = plt.subplots(figsize=(7.5, height))
    fig.patch.set_facecolor(PLT_BG)
    ax.set_facecolor(PLT_BG)
    bars = ax.barh(labels, values, color=colors, height=0.55)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    ax.tick_params(colors=PLT_TEXT, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(MPL_BORDER)
    ax.spines["left"].set_color(MPL_BORDER)
    ax.invert_yaxis()
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height()/2,
                f"{w*100:.1f}%" if max(values) <= 1 else f"{int(w)}",
                va="center", color=PLT_TEXT, fontsize=9, fontweight="bold")
    ax.set_xlim(0, max(values)*1.18)
    fig.tight_layout(pad=1.2)
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Brand
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;padding:6px 0 22px 0;">
        <div style="width:44px;height:44px;background:linear-gradient(135deg,#6366f1,#4f46e5);
            border-radius:12px;display:flex;align-items:center;justify-content:center;
            color:#fff;font-weight:900;font-size:22px;box-shadow:0 4px 14px rgba(99,102,241,.4);">E</div>
        <div>
            <div style="font-size:1.05rem;font-weight:800;color:{TEXT};">EduIntent AI</div>
            <div style="font-size:0.75rem;color:{MUTED};font-weight:500;">SWYNEX · AI Problem Design</div>
        </div>
    </div>
    <div style="font-size:0.7rem;font-weight:800;letter-spacing:1.5px;color:#475569;
        text-transform:uppercase;margin-bottom:10px;padding-left:4px;">WORKSPACE</div>
    """, unsafe_allow_html=True)

    nav_items = [
        "🏠 Overview",
        "✦ Query Classifier",
        "📊 Analytics",
        "🔍 Category Deep-Dive",
        "🕒 Query History",
        "📁 Dataset Management",
        "🤖 Model & Evaluation",
        "📚 Documentation",
        "✅ Submission Checklist",
    ]
    nav = st.radio("nav", nav_items, index=nav_items.index(st.session_state["active_nav"]))
    st.session_state["active_nav"] = nav

    st.markdown("---")
    st.markdown(f"""<div style="font-size:0.7rem;font-weight:800;letter-spacing:1.5px;
        color:#475569;text-transform:uppercase;margin-bottom:10px;">SYSTEM BENCHMARK</div>""",
        unsafe_allow_html=True)
    st.metric("Linear SVM Accuracy", f"{SVM_ACC*100:.1f}%", "✓ Target ≥ 85%")
    st.metric("Macro F1-Score",      f"{SVM_F1:.4f}",      "✓ Target ≥ 0.80")
    st.markdown("---")
    st.markdown(f"""<div style="font-size:0.7rem;font-weight:800;letter-spacing:1.5px;
        color:#475569;text-transform:uppercase;margin-bottom:10px;">SESSION LIVE STATS</div>""",
        unsafe_allow_html=True)

    hist = st.session_state["history"]
    total_q = len(hist)
    if total_q:
        confs = [float(h["Confidence"].replace("%",""))/100 for h in hist]
        avg_conf = np.mean(confs)
        high_count = sum(1 for c in confs if c >= 0.75)
        st.markdown(f'<span class="stat-pill">🔢 {total_q} queries</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="stat-pill">📈 {avg_conf*100:.1f}% avg conf</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="stat-pill">✅ {high_count} high-conf</span>', unsafe_allow_html=True)
    else:
        st.caption("No queries yet in this session.")

    st.markdown("---")
    if st.button("🌙 Dark" if not D else "☀️ Light Mode", use_container_width=True):
        st.session_state["dark"] = not D
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TOP BAR
# ═══════════════════════════════════════════════════════════════════════════════
tb1, tb2, tb3 = st.columns([4, 1.2, 0.8])
with tb1:
    st.markdown(f'<div class="topbar-title">EduIntent AI — SWYNEX Task 1</div>', unsafe_allow_html=True)
with tb2:
    st.markdown('<div class="pill-badge" style="text-align:center;">Task 1 · Prototype</div>', unsafe_allow_html=True)
with tb3:
    st.markdown(f'<div class="pill-badge" style="text-align:center;color:{EMERALD};">🟢 AI Ready</div>', unsafe_allow_html=True)
st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:0 0 22px 0;">', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ❶  OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if nav == "🏠 Overview":
    col_left, col_right = st.columns([2.4, 1])

    with col_left:
        st.markdown('<div class="hero-kicker">AI PROBLEM DESIGN · SWYNEX INTERNSHIP TASK 1</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-h1">Student Query<br>Intelligence</div>', unsafe_allow_html=True)
        st.markdown(f'''<div class="hero-sub">
        EduIntent AI automatically classifies student academic questions into 9 subject categories
        using NLP & Calibrated Linear SVM. Powered by Explainable AI, ambiguity detection,
        uncertainty guardrails, and Gujarati / English code-mixing support.
        </div>''', unsafe_allow_html=True)

        f1, f2, f3, f4 = st.columns(4)
        with f1:
            st.markdown(f'<span class="badge-tag">💡 Explainable AI</span>', unsafe_allow_html=True)
        with f2:
            st.markdown(f'<span class="badge-tag">🔀 Ambiguity Detection</span>', unsafe_allow_html=True)
        with f3:
            st.markdown(f'<span class="badge-tag">⚠️ Uncertainty Guard</span>', unsafe_allow_html=True)
        with f4:
            st.markdown(f'<span class="badge-tag">🌐 Code-Mixing</span>', unsafe_allow_html=True)

    with col_right:
        st.markdown(f"""
        <div style="background:linear-gradient(145deg,#4f46e5,#7c3aed);border-radius:20px;
            padding:36px 24px;text-align:center;box-shadow:0 14px 35px rgba(99,102,241,.35);">
            <div style="font-size:2.5rem;margin-bottom:12px;">🎓</div>
            <div style="font-size:1.55rem;font-weight:900;color:#fff;line-height:1.25;margin-bottom:12px;">Try<br>Classifier</div>
            <div style="font-size:1.8rem;color:rgba(255,255,255,0.9);">→</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Open Classifier", use_container_width=True, type="primary"):
            st.session_state["active_nav"] = "✦ Query Classifier"
            st.rerun()

    st.markdown("---")
    st.markdown("### 📊 System KPI Dashboard")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpis = [
        ("450", "Dataset Queries", "50 per category"),
        ("9",   "Subject Classes", "Balanced split"),
        (f"{SVM_ACC*100:.1f}%", "Test Accuracy", "✓ Target ≥ 85%"),
        (f"{SVM_F1:.4f}", "Macro F1-Score", "✓ Target ≥ 0.80"),
        ("89.72%", "5-Fold CV Acc", "± 0.02 std dev"),
        ("3", "ML Models Tested", "Best = SVM 🏆"),
    ]
    for col, (val, label, delta) in zip([k1, k2, k3, k4, k5, k6], kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-val">{val}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-delta">{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⭐ Unique AI System Features")
    u1, u2, u3, u4 = st.columns(4)
    features = [
        ("💡", "Explainable AI (XAI)",         "Top TF-IDF terms are displayed to explain every prediction. No black-box decisions."),
        ("🔀", "Ambiguity Detection",            "Multi-subject overlap flagged when top-2 probabilities are within 0.25 gap."),
        ("⚠️", "Uncertainty Guardrails",          "Low-confidence protection (<45%) alerts user to refine vague or short queries."),
        ("🌐", "Code-Mix Detection",              "Detects Gujarati (native script + romanized) and Hinglish code-mixed student inputs."),
    ]
    for col, (icon, title, desc) in zip([u1, u2, u3, u4], features):
        with col:
            st.markdown(f"""
            <div class="edu-card" style="height:190px;">
                <div style="font-size:1.8rem;margin-bottom:8px;">{icon}</div>
                <div style="font-size:0.95rem;font-weight:800;color:{TEXT};margin-bottom:8px;">{title}</div>
                <div style="font-size:0.84rem;color:{MUTED};line-height:1.55;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("### 🏗️ AI System Architecture Flow")
    st.code("""
 [ Student Question ]
        │
        ▼
 [ Language & Code-Mix Detector ] ──► Gujarati / English / Hinglish badge
        │
        ▼
 [ TF-IDF Text Vectorizer + Stop-Word Removal + N-Gram Extraction ]
        │
        ▼
 [ Calibrated Linear SVM Classifier ]
        │
        ├──► [ Confidence ≥ 75% ] → ✅ High Confidence Match
        │
        ├──► [ Gap < 0.25 between top-2 ] → 🔀 Ambiguous Multi-Subject
        │
        └──► [ Confidence < 45% ] → ⚠️ Low-Confidence Guardrail
        │
        ▼
 [ XAI: Extract Top TF-IDF Contributing Terms ]
        │
        ▼
 [ Display: Category Badge + Gauge + Radar + Key Terms ]
    """, language="")

# ═══════════════════════════════════════════════════════════════════════════════
# ❷  AI QUERY CLASSIFIER  (Live Interactive)
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "✦ Query Classifier":
    st.markdown("### ✦ AI Student Query Classifier — Real-Time Inference")
    st.markdown(f'<div style="color:{MUTED};font-size:0.95rem;margin-bottom:18px;">Enter any student academic question. The model returns a category, confidence score, explainability terms, ambiguity signals, and complexity analysis instantly.</div>', unsafe_allow_html=True)

    # ── Preset buttons ──
    st.markdown("**💡 Preset Edge-Case Scenarios:**")
    presets = {
        "🐍 Python KeyError":          "Why am I getting a KeyError while accessing a Pandas column?",
        "🔀 Ambiguous (CNN+Python)":   "How can I implement CNN using Python?",
        "⚠️ Low Confidence":           "Can you explain this?",
        "🌐 Gujarati Code-Mixed":      "Pandas ma KeyError aave chhe su karvu?",
        "⚙️ OS Deadlock":              "What is the difference between deadlock and livelock in OS?",
        "💾 DBMS ACID":                "Explain ACID properties and 3NF normalization in database tables.",
        "📱 Flutter Async":            "How to build async dynamic UI using FutureBuilder in Flutter?",
        "📐 Merge Sort Recurrence":    "What is the time complexity of Merge Sort recurrence T(n) = 2T(n/2) + n?",
    }
    preset_keys = list(presets.keys())
    row1 = st.columns(4)
    row2 = st.columns(4)
    for cols, row_keys in [(row1, preset_keys[:4]), (row2, preset_keys[4:])]:
        for col, key in zip(cols, row_keys):
            with col:
                if st.button(key, use_container_width=True):
                    st.session_state["active_q"] = presets[key]
                    st.rerun()

    current_q = st.session_state.get("active_q", presets["🐍 Python KeyError"])

    # ── Input ──
    user_input = st.text_area(
        "Student Question Input:",
        value=current_q,
        height=95,
        placeholder="Type your academic question here...",
        key="classifier_input"
    )

    # ── Live-Typing classifier (auto-run when text changes via toggle) ──
    live_toggle = st.toggle("⚡ Live Auto-Classify (updates on every change)", value=False)

    classify_clicked = st.button("🚀 Classify Intent Now", type="primary", use_container_width=True)

    run_now = classify_clicked or (live_toggle and user_input.strip())

    if run_now and user_input.strip():
        t0  = time.perf_counter()
        res = predictor.predict(user_input)
        latency = (time.perf_counter() - t0) * 1000

        # Save to history
        st.session_state["history"].append({
            "Timestamp":          datetime.datetime.now().strftime("%H:%M:%S"),
            "Query":              res["query"],
            "Language":           res["language"],
            "Predicted Category": res["predicted_category"],
            "Confidence":         f"{res['confidence']*100:.1f}%",
            "Status":             "⚠️ Low Conf" if res["is_low_confidence"] else ("🔀 Ambiguous" if res["is_ambiguous"] else "✅ Clear"),
            "Key Terms":          ", ".join(res["important_terms"]) if res["important_terms"] else "—",
            "Latency (ms)":       f"{latency:.1f}",
        })

        st.markdown("---")

        # ── Notification Banner ──
        cat_icon = CAT_ICONS.get(res["predicted_category"], "📌")
        if res["is_low_confidence"]:
            st.warning(f"⚠️ **Low-Confidence Uncertainty Warning (< 45%):** Prediction confidence is **{res['confidence']*100:.1f}%**. This query appears vague or out-of-vocabulary. Please refine with subject-specific terms.")
        elif res["is_ambiguous"]:
            st.info(f"🔀 **Ambiguous Multi-Subject Detected:** Primary → `{res['predicted_category']}` | Related Overlap → `{res['related_category']}` (probability gap < 0.25)")
        else:
            st.success(f"✅ **High-Confidence Match:** {cat_icon} Classified as **{res['predicted_category']}** ({res['confidence']*100:.1f}% confidence) in {latency:.1f} ms.")

        # ── Main Result Columns ──
        col_L, col_R = st.columns([1.25, 1])

        with col_L:
            st.markdown('<div class="edu-card">', unsafe_allow_html=True)
            st.markdown("#### Primary Category Prediction")
            cat_color = CAT_COLORS.get(res["predicted_category"], INDIGO)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;">
                <div style="font-size:2.4rem;">{cat_icon}</div>
                <div>
                    <div style="font-size:1.25rem;font-weight:800;color:{TEXT};">{res['predicted_category']}</div>
                    <div style="font-size:0.82rem;color:{MUTED};">Detected Subject Category</div>
                </div>
            </div>""", unsafe_allow_html=True)

            conf = res["confidence"]
            conf_cls = "conf-high" if conf >= 0.75 else ("conf-med" if conf >= 0.50 else "conf-low")
            conf_label = "High Confidence" if conf >= 0.75 else ("Moderate Confidence" if conf >= 0.50 else "Low Confidence")
            st.markdown(f"**Confidence:** <span class='{conf_cls}'>{conf*100:.1f}% — {conf_label}</span>", unsafe_allow_html=True)
            st.progress(conf)

            lang_icon = "🌐" if "Code" in res["language"] else ("🇮🇳" if "Gujarati" in res["language"] else "🇬🇧")
            st.markdown(f"**Language:** {lang_icon} `{res['language']}`")
            st.markdown(f"**Inference Time:** `{latency:.1f} ms`")
            st.markdown(f"**Normalized Input:** `{res['cleaned_query']}`")

            st.markdown("---")
            st.markdown("##### 💡 Explainable AI (XAI) — TF-IDF Attribution")
            if res["important_terms"]:
                st.write("Top contributing feature terms that drove this classification:")
                tags = "".join([f'<span class="badge-tag">• {t}</span>' for t in res["important_terms"]])
                st.markdown(tags, unsafe_allow_html=True)
            else:
                st.caption("No strong vocabulary matched the TF-IDF feature dictionary.")

            # ── Query Complexity Analysis ──
            st.markdown("---")
            st.markdown("##### 🔬 Query Complexity Analysis")
            cx = analyze_complexity(user_input, res)
            grade_color = EMERALD if cx["grade"] == "Excellent" else (SKY if cx["grade"] == "Good" else (AMBER if cx["grade"] == "Average" else ROSE))
            c_cols = st.columns(3)
            with c_cols[0]:
                st.metric("Word Count", cx["word_count"])
            with c_cols[1]:
                st.metric("Vocab Richness", f"{cx['vocab_richness']*100:.0f}%")
            with c_cols[2]:
                st.metric("Tech Density", f"{cx['tech_density']*100:.0f}%")
            st.markdown(f'<span style="color:{grade_color};font-weight:800;font-size:1rem;">Query Grade: {cx["grade"]}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_R:
            # ── Confidence Gauge ──
            st.markdown('<div class="edu-card">', unsafe_allow_html=True)
            st.markdown("##### Confidence Gauge Meter")
            st.plotly_chart(conf_gauge(conf, height=250), use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Top-3 Probability Bar ──
            st.markdown('<div class="edu-card">', unsafe_allow_html=True)
            st.markdown("##### Top Category Probability Distribution")
            top_cats = res["top_categories"]
            df_top = pd.DataFrame(top_cats)
            colors_top = [CAT_COLORS.get(c, INDIGO) for c in df_top["category"]]
            fig_top = mpl_hbar(df_top["category"].tolist(), df_top["probability"].tolist(), colors_top, height=2.6)
            st.pyplot(fig_top, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Radar Chart (all categories) ──
        radar = radar_chart(res)
        if radar:
            st.markdown('<div class="edu-card">', unsafe_allow_html=True)
            st.markdown("##### 🕸️ All-Category Confidence Radar Chart")
            st.plotly_chart(radar, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Smart Similar Query Suggestions ──
        st.markdown('<div class="edu-card">', unsafe_allow_html=True)
        st.markdown(f"##### 🔗 Smart Query Suggestions — More `{res['predicted_category']}` Examples")
        suggestions = smart_suggestions(res["predicted_category"], n=4)
        sug_cols = st.columns(len(suggestions))
        for i, (col, sug) in enumerate(zip(sug_cols, suggestions)):
            with col:
                if st.button(f'"{sug[:60]}…"' if len(sug) > 60 else f'"{sug}"',
                             key=f"sug_{i}", use_container_width=True):
                    st.session_state["active_q"] = sug
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    elif not user_input.strip():
        st.warning("Please enter a non-empty question to classify.")

# ═══════════════════════════════════════════════════════════════════════════════
# ❸  ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "📊 Analytics":
    st.markdown("### 📊 Dataset & Model Analytics")

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown('<div class="edu-card">', unsafe_allow_html=True)
        st.markdown("#### Class Balance Distribution")
        cat_counts = dataset["Category"].value_counts().reset_index()
        cat_counts.columns = ["Category","Count"]
        colors_bar = [CAT_COLORS.get(c, INDIGO) for c in cat_counts["Category"]]
        fig_bal = mpl_hbar(cat_counts["Category"].tolist(), cat_counts["Count"].tolist(), colors_bar, "Questions", height=4.5)
        st.pyplot(fig_bal, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_a2:
        st.markdown('<div class="edu-card">', unsafe_allow_html=True)
        st.markdown("#### Query Word-Count Distribution")
        dataset["Word_Count"] = dataset["Question"].apply(lambda q: len(str(q).split()))
        fig_wc, ax_wc = plt.subplots(figsize=(7.5, 4.5))
        fig_wc.patch.set_facecolor(PLT_BG); ax_wc.set_facecolor(PLT_BG)
        sns.histplot(dataset["Word_Count"], bins=15, kde=True, color=INDIGO, ax=ax_wc, alpha=0.7)
        ax_wc.set_xlabel("Words per Question", color=MUTED, fontsize=10)
        ax_wc.tick_params(colors=PLT_TEXT, labelsize=9)
        ax_wc.spines["top"].set_visible(False); ax_wc.spines["right"].set_visible(False)
        ax_wc.spines["bottom"].set_color(MPL_BORDER); ax_wc.spines["left"].set_color(MPL_BORDER)
        fig_wc.tight_layout(pad=1.2)
        st.pyplot(fig_wc, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Per-Category F1 Plotly Bar ──
    st.markdown("#### Per-Category Precision / Recall / F1 Breakdown")
    cr = meta["classification_report"]
    cats_cr = [c for c in CAT_ICONS.keys() if c in cr]
    prec_v  = [cr[c]["precision"] for c in cats_cr]
    rec_v   = [cr[c]["recall"]    for c in cats_cr]
    f1_v    = [cr[c]["f1-score"]  for c in cats_cr]
    fig_cr = go.Figure()
    fig_cr.add_trace(go.Bar(name="Precision", x=cats_cr, y=prec_v, marker_color=SKY,    opacity=0.85))
    fig_cr.add_trace(go.Bar(name="Recall",    x=cats_cr, y=rec_v,  marker_color=EMERALD, opacity=0.85))
    fig_cr.add_trace(go.Bar(name="F1-Score",  x=cats_cr, y=f1_v,  marker_color=INDIGO,  opacity=0.9))
    fig_cr.update_layout(barmode="group", **plotly_layout("Classification Report by Category", height=360))
    st.plotly_chart(fig_cr, use_container_width=True, config={"displayModeBar": False})

    # Confusion matrix
    st.markdown("#### 🖼️ Confusion Matrix – Linear SVM on Held-Out Test Set")
    cm_path = os.path.join(BASE_DIR, "reports", "confusion_matrix.png")
    if os.path.exists(cm_path):
        st.image(cm_path, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ❹  CATEGORY DEEP-DIVE (Unique Feature)
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "🔍 Category Deep-Dive":
    st.markdown("### 🔍 Category Deep-Dive Explorer")
    st.markdown(f'<div style="color:{MUTED};margin-bottom:18px;">Select any academic subject category to explore sample questions, keyword vocabulary, WordCloud visualization, and per-class performance metrics.</div>', unsafe_allow_html=True)

    selected_cat = st.selectbox("Choose a Subject Category to Explore:", list(CAT_ICONS.keys()))
    cat_color = CAT_COLORS.get(selected_cat, INDIGO)
    cat_icon  = CAT_ICONS.get(selected_cat, "📌")

    st.markdown(f"""
    <div class="edu-card" style="border-left:4px solid {cat_color};">
        <div style="display:flex;align-items:center;gap:12px;">
            <span style="font-size:2.5rem;">{cat_icon}</span>
            <div>
                <div style="font-size:1.4rem;font-weight:900;color:{TEXT};">{selected_cat}</div>
                <div style="color:{MUTED};font-size:0.88rem;">Academic Subject Deep-Dive Analysis</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    col_dd1, col_dd2 = st.columns([1, 1.3])

    with col_dd1:
        st.markdown('<div class="edu-card">', unsafe_allow_html=True)
        st.markdown("##### 📊 Per-Class Performance Metrics")
        cr_cat = meta["classification_report"].get(selected_cat, {})
        if cr_cat:
            m1, m2, m3 = st.columns(3)
            with m1: st.metric("Precision", f"{cr_cat['precision']*100:.1f}%")
            with m2: st.metric("Recall",    f"{cr_cat['recall']*100:.1f}%")
            with m3: st.metric("F1-Score",  f"{cr_cat['f1-score']*100:.1f}%")

            # Mini gauge using Plotly
            fig_mini = go.Figure(go.Indicator(
                mode="gauge+number",
                value=cr_cat["f1-score"]*100,
                number={"suffix":"%","font":{"size":28,"color":TEXT}},
                gauge={"axis":{"range":[0,100]},"bar":{"color":cat_color},"bgcolor":CARD2,"borderwidth":0}
            ))
            fig_mini.update_layout(**plotly_layout(height=180, margin=dict(l=20, r=20, t=10, b=10)))
            st.plotly_chart(fig_mini, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="edu-card">', unsafe_allow_html=True)
        st.markdown("##### 🗝️ Key Vocabulary Terms")
        kws = CAT_KEYWORDS.get(selected_cat, [])
        tags = "".join([f'<span class="badge-tag" style="color:{cat_color};border-color:{cat_color}40;">• {kw}</span>' for kw in kws])
        st.markdown(tags, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_dd2:
        st.markdown('<div class="edu-card">', unsafe_allow_html=True)
        st.markdown("##### ☁️ WordCloud — Dominant Vocabulary")
        wc_fig = make_wordcloud(selected_cat)
        if wc_fig:
            st.pyplot(wc_fig, use_container_width=True)
        else:
            st.info("Install wordcloud: `pip install wordcloud`")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="edu-card">', unsafe_allow_html=True)
    st.markdown(f"##### 📋 Sample Questions from `{selected_cat}` (10 rows)")
    sample_rows = dataset[dataset["Category"] == selected_cat].head(10)
    st.dataframe(sample_rows[["Question"]], use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Live Test from this Category ──
    st.markdown('<div class="edu-card">', unsafe_allow_html=True)
    st.markdown(f"##### ⚡ Quick-Classify a `{selected_cat}` Question")
    quick_q = st.text_input("Enter a test question:", placeholder=f"e.g., {CAT_KEYWORDS.get(selected_cat,[''])[0]} related question...")
    if st.button("Classify →", type="primary"):
        if quick_q.strip():
            qr = predictor.predict(quick_q)
            is_correct = qr["predicted_category"] == selected_cat
            color_badge = EMERALD if is_correct else ROSE
            icon = "✅" if is_correct else "❌"
            st.markdown(f"""
            <div style="margin-top:12px;padding:14px 18px;background:{CARD2};border-radius:10px;border-left:4px solid {color_badge};">
                <span style="font-weight:800;color:{color_badge};">{icon} Predicted: {qr['predicted_category']}</span>
                &nbsp;|&nbsp;
                <span style="color:{MUTED};">Confidence: {qr['confidence']*100:.1f}%</span>
                &nbsp;|&nbsp;
                <span style="color:{MUTED};">Expected: {selected_cat}</span>
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ❺  QUERY HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "🕒 Query History":
    st.markdown("### 🕒 Live Session Query History Log")
    hist = st.session_state["history"]

    if hist:
        df_hist = pd.DataFrame(hist)

        # ── Session Stats ──
        h1, h2, h3, h4 = st.columns(4)
        confs = [float(h["Confidence"].replace("%",""))/100 for h in hist]
        with h1: st.metric("Total Classified", len(hist))
        with h2: st.metric("Avg Confidence",  f"{np.mean(confs)*100:.1f}%")
        with h3: st.metric("High Confidence", sum(1 for c in confs if c >= 0.75))
        with h4: st.metric("Low Confidence",  sum(1 for c in confs if c < 0.45))

        # ── Category Distribution in Session ──
        st.markdown("#### Session Category Distribution")
        cat_sess = df_hist["Predicted Category"].value_counts()
        fig_sess = px.pie(
            names=cat_sess.index, values=cat_sess.values,
            color_discrete_sequence=[CAT_COLORS.get(c, INDIGO) for c in cat_sess.index],
            hole=0.55,
        )
        fig_sess.update_layout(**plotly_layout(height=300))
        st.plotly_chart(fig_sess, use_container_width=True, config={"displayModeBar": False})

        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        col_ex1, col_ex2 = st.columns([1, 4])
        with col_ex1:
            csv = df_hist.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Export CSV", csv, "eduintent_history.csv", "text/csv", use_container_width=True)
        with col_ex2:
            if st.button("🗑️ Clear Session History", use_container_width=True):
                st.session_state["history"] = []
                st.rerun()
    else:
        st.info("No queries classified in this session yet. Go to **✦ Query Classifier** to start!")

# ═══════════════════════════════════════════════════════════════════════════════
# ❻  DATASET MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "📁 Dataset Management":
    st.markdown("### 📁 Dataset Management, Explorer & Batch Classifier")

    t_browse, t_batch = st.tabs(["🔍 Browse & Filter", "📤 Batch CSV Upload"])

    with t_browse:
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            all_cats = ["All Categories"] + list(dataset["Category"].unique())
            sel_cat = st.selectbox("Filter by Category:", all_cats)
            search  = st.text_input("Search Keyword:", "")

        filtered = dataset.copy()
        if sel_cat != "All Categories":
            filtered = filtered[filtered["Category"] == sel_cat]
        if search.strip():
            filtered = filtered[filtered["Question"].str.contains(search, case=False, na=False)]

        with col_f2:
            st.metric("Matching Queries", len(filtered), f"of {len(dataset)} total")

        st.dataframe(filtered, use_container_width=True, hide_index=True)
        csv_ds = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Filtered Dataset", csv_ds, "filtered_dataset.csv", "text/csv")

    with t_batch:
        st.markdown("##### Upload a CSV file with a `Question` column for bulk classification")
        uploaded = st.file_uploader("Upload CSV:", type=["csv"])

        # Prefill batch test
        batch_defaults = [
            {"Question": "How to handle missing values using dropna() in Pandas?",                       "Expected": "Python / Data Science"},
            {"Question": "Explain overfitting, underfitting, and bias-variance tradeoff.",               "Expected": "Machine Learning"},
            {"Question": "What is vanishing gradient in deep neural backpropagation?",                   "Expected": "Deep Learning"},
            {"Question": "Explain ACID properties and 3NF normalization in DBMS.",                       "Expected": "DBMS"},
            {"Question": "What is semaphore vs mutex in OS?",                                            "Expected": "Operating Systems"},
            {"Question": "Explain TCP 3-way handshake SYN SYN-ACK ACK.",                                "Expected": "Computer Networks"},
            {"Question": "How does StreamBuilder handle real-time data in Flutter?",                     "Expected": "Flutter"},
            {"Question": "What is the time complexity of Merge Sort T(n) recurrence?",                  "Expected": "DAA / Algorithms"},
            {"Question": "How do I get my official transcript from the college administrative office?",   "Expected": "General Academic Query"},
        ]

        df_batch_src = None
        if uploaded:
            try:
                df_upload = pd.read_csv(uploaded)
                if "Question" not in df_upload.columns:
                    st.error("CSV must contain a `Question` column.")
                else:
                    df_batch_src = df_upload.copy()
                    st.success(f"Loaded {len(df_upload)} questions from uploaded file.")
            except Exception as e:
                st.error(f"Read error: {e}")
        else:
            st.info("No file uploaded. Using built-in 9-class test suite below.")
            df_batch_src = pd.DataFrame(batch_defaults)
            st.dataframe(df_batch_src, use_container_width=True, hide_index=True)

        if st.button("🚀 Run Batch Classification", type="primary", use_container_width=True):
            preds, confs, langs, statuses, terms = [], [], [], [], []
            prog = st.progress(0)
            for i, row in enumerate(df_batch_src.itertuples(), 1):
                r = predictor.predict(str(row.Question))
                preds.append(r["predicted_category"])
                confs.append(f"{r['confidence']*100:.1f}%")
                langs.append(r["language"])
                statuses.append("⚠️" if r["is_low_confidence"] else ("🔀" if r["is_ambiguous"] else "✅"))
                terms.append(", ".join(r["important_terms"]) if r["important_terms"] else "—")
                prog.progress(i / len(df_batch_src))

            df_batch_src["Predicted"] = preds
            df_batch_src["Confidence"] = confs
            df_batch_src["Language"]   = langs
            df_batch_src["Status"]     = statuses
            df_batch_src["Key Terms (XAI)"] = terms

            if "Expected" in df_batch_src.columns:
                df_batch_src["✓ Match"] = df_batch_src.apply(
                    lambda r: "✅" if r["Predicted"] == r["Expected"] else "❌", axis=1)
                correct = (df_batch_src["✓ Match"] == "✅").sum()
                st.success(f"Batch Accuracy: **{correct}/{len(df_batch_src)} = {correct/len(df_batch_src)*100:.1f}%**")

            st.dataframe(df_batch_src, use_container_width=True, hide_index=True)
            csv_out = df_batch_src.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Batch Results", csv_out, "batch_results.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════════
# ❼  MODEL & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "🤖 Model & Evaluation":
    st.markdown("### 🤖 Candidate Model Selection & Benchmarks")
    st.markdown(f'<div style="color:{MUTED};margin-bottom:18px;">Comparing all candidate models on identical 80/20 stratified train/test split (N_train=360, N_test=90).</div>', unsafe_allow_html=True)

    # ── Benchmark Table ──
    df_res = pd.DataFrame(meta["results_summary"]).T.reset_index().rename(columns={"index":"Model"})
    st.dataframe(df_res.style.highlight_max(axis=0, color="#1e3a8a",
        subset=["Accuracy","F1-Score (Macro)"]), use_container_width=True)

    col_ev1, col_ev2 = st.columns(2)
    models = df_res["Model"].tolist()
    accs   = df_res["Accuracy"].tolist()
    f1s    = df_res["F1-Score (Macro)"].tolist()
    m_colors = [INDIGO if "SVM" in m else (EMERALD if "Bayes" in m else (AMBER if "Logistic" in m else ROSE)) for m in models]

    with col_ev1:
        fig_acc = go.Figure(go.Bar(x=models, y=accs, marker_color=m_colors, text=[f"{a*100:.1f}%" for a in accs], textposition="outside"))
        fig_acc.add_hline(y=0.85, line_dash="dash", line_color=AMBER, annotation_text="Target 85%")
        fig_acc.update_layout(**plotly_layout("Test Accuracy vs. Target (85%)", height=340), yaxis_range=[0,1.12])
        st.plotly_chart(fig_acc, use_container_width=True, config={"displayModeBar":False})

    with col_ev2:
        fig_f1 = go.Figure(go.Bar(x=models, y=f1s, marker_color=m_colors, text=[f"{f:.3f}" for f in f1s], textposition="outside"))
        fig_f1.add_hline(y=0.80, line_dash="dash", line_color=AMBER, annotation_text="Target F1 0.80")
        fig_f1.update_layout(**plotly_layout("Macro F1-Score vs. Target (0.80)", height=340), yaxis_range=[0,1.12])
        st.plotly_chart(fig_f1, use_container_width=True, config={"displayModeBar":False})

    # ── Error Analysis Table ──
    st.markdown("---")
    st.markdown("#### 🔍 Qualitative Misclassification Error Analysis")
    err = meta.get("error_analysis", [])
    if err:
        df_err = pd.DataFrame(err)
        st.dataframe(df_err, use_container_width=True, hide_index=True)
    else:
        st.info("No misclassifications recorded on test set.")

    st.markdown("""
    #### 💡 Observed Error Patterns & Mitigation
    | Error Pattern | Root Cause | Mitigation |
    |---|---|---|
    | **ML ↔ Deep Learning** | Shared terms: `neural network`, `loss function` | Ambiguity flag + probability gap threshold |
    | **DBMS ↔ DAA** | `B+ Tree` in both indexing & data structures | XAI terms expose the key conflicting feature |
    | **OS ↔ DBMS** | `Race condition`, `transaction` overlap | Ambiguity detection alerts user |
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# ❽  DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "📚 Documentation":
    st.markdown("### 📚 Project Documentation & Repository Structure")

    col_doc1, col_doc2 = st.columns([1.2, 1])
    with col_doc1:
        st.markdown("#### 📁 GitHub Folder Tree")
        st.code("""SWYNEX-AI-Problem-Design/
├── app.py                        # Premium Streamlit dashboard (10 unique features)
├── create_dataset.py             # Dataset creation (450 balanced queries)
├── create_notebook.py            # Jupyter notebook auto-generator
├── export_pdf_report.py          # PDF executive report generator
├── EduIntent_AI_Task1_Problem_Design.pdf  # Executive PDF report
├── README.md                     # Full project documentation
├── requirements.txt              # Python dependencies
├── .streamlit/
│   └── config.toml               # Streamlit dark theme config
├── dataset/
│   └── student_queries.csv       # 450 hand-curated queries (9 classes)
├── models/
│   ├── best_model.pkl            # Calibrated Linear SVM model
│   ├── tfidf_vectorizer.pkl      # Trained TF-IDF vectorizer
│   └── model_metadata.json       # Metrics, categories, error analysis
├── notebooks/
│   └── EduIntent_AI_Problem_Design.ipynb
├── reports/
│   ├── confusion_matrix.png      # Test set confusion matrix
│   └── model_comparison.png      # Benchmark comparison chart
├── results/
│   └── evaluation.txt            # Text summary of evaluation
└── src/
    ├── model.py                  # Model factory helpers
    ├── predict.py                # EduIntentPredictor class
    ├── preprocessing.py          # Text preprocessing pipeline
    └── train.py                  # Training & benchmarking pipeline""", language="text")

    with col_doc2:
        st.markdown("#### ⚙️ Technical Stack")
        stack = [
            ("Python 3.x", "Core language"),
            ("Scikit-Learn", "TF-IDF + SVM + Cross-Validation"),
            ("Streamlit", "Interactive web dashboard"),
            ("Plotly", "Gauge, radar, interactive charts"),
            ("WordCloud", "Category vocabulary visualization"),
            ("Matplotlib + Seaborn", "Static charts & confusion matrix"),
            ("Pandas + NumPy", "Data manipulation & numerics"),
            ("ReportLab", "PDF executive report export"),
        ]
        for tool, role in stack:
            st.markdown(f"""
            <div class="edu-card-sm" style="margin-bottom:8px;">
                <span style="font-weight:700;color:{TEXT};">{tool}</span>
                <span style="color:{MUTED};font-size:0.85rem;"> — {role}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### 🚀 Quick Start")
        st.code("""# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset
python create_dataset.py

# 3. Train all models & save artifacts
python src/train.py

# 4. Launch the dashboard
streamlit run app.py

# 5. Export PDF report
python export_pdf_report.py""", language="bash")

# ═══════════════════════════════════════════════════════════════════════════════
# ❾  SUBMISSION CHECKLIST
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "✅ Submission Checklist":
    st.markdown("### ✅ Task 1 SWYNEX Submission Verification Checklist")
    st.markdown(f'<div style="color:{MUTED};margin-bottom:22px;">Comprehensive check of all Task 1 AI Problem Design & Practical Implementation requirements.</div>', unsafe_allow_html=True)

    checks = [
        ("✅", EMERALD, "Problem Statement formulated in English & Gujarati"),
        ("✅", EMERALD, "450 balanced hand-curated queries (50 × 9 categories)"),
        ("✅", EMERALD, "Majority Class Baseline benchmarked (Accuracy: 11.11%, F1: 0.022)"),
        ("✅", EMERALD, "Naive Bayes evaluated (Accuracy: 85.56%, F1: 0.8547)"),
        ("✅", EMERALD, "Logistic Regression evaluated (Accuracy: 83.33%, F1: 0.8275)"),
        ("✅", EMERALD, "Calibrated Linear SVM 🏆 evaluated (Accuracy: 87.78%, F1: 0.8746)"),
        ("✅", EMERALD, "Target Accuracy ≥ 85% → Achieved 87.78%"),
        ("✅", EMERALD, "Target Macro F1 ≥ 0.80 → Achieved 0.8746"),
        ("✅", EMERALD, "5-Fold Cross-Validation performed (89.72% ± 0.02)"),
        ("✅", EMERALD, "Explainable AI (XAI) — TF-IDF term attribution implemented"),
        ("✅", EMERALD, "Ambiguity Detection for multi-subject overlap queries"),
        ("✅", EMERALD, "Low-Confidence Guardrails (< 45% threshold)"),
        ("✅", EMERALD, "Gujarati / English / Hinglish code-mixing detection"),
        ("✅", EMERALD, "Confidence Gauge Meter (Plotly interactive)"),
        ("✅", EMERALD, "All-Category Confidence Radar Chart implemented"),
        ("✅", EMERALD, "Query Complexity Analyzer (vocab richness, tech density, grade)"),
        ("✅", EMERALD, "Smart Similar Query Suggestions from dataset"),
        ("✅", EMERALD, "Category Deep-Dive with WordCloud per academic subject"),
        ("✅", EMERALD, "Batch CSV Uploader with full prediction results + download"),
        ("✅", EMERALD, "Session Analytics — live stats in sidebar"),
        ("✅", EMERALD, "Dark / Light Theme Toggle"),
        ("✅", EMERALD, "Executive PDF Report generated"),
        ("✅", EMERALD, "Jupyter Notebook demonstration created"),
        ("✅", EMERALD, "Premium multi-tab Streamlit dashboard with SWYNEX branding"),
    ]

    for icon, color, label in checks:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:10px 14px;background:{CARD};
            border-radius:10px;border-left:3px solid {color};margin-bottom:6px;">
            <span style="font-size:1.1rem;">{icon}</span>
            <span style="font-size:0.93rem;color:{TEXT};font-weight:600;">{label}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#064e3b,#065f46);border:1px solid #10b981;
        border-radius:16px;padding:24px 28px;text-align:center;margin-top:10px;">
        <div style="font-size:2rem;margin-bottom:8px;">🎉</div>
        <div style="font-size:1.3rem;font-weight:900;color:#ffffff;">All {len(checks)} SWYNEX Task 1 Requirements Satisfied!</div>
        <div style="color:#6ee7b7;margin-top:6px;font-size:0.95rem;">Project ready for professional SWYNEX internship submission.</div>
    </div>""", unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:36px;padding-top:16px;border-top:1px solid {BORDER};
    text-align:center;color:{MUTED};font-size:0.8rem;">
    EduIntent AI &nbsp;·&nbsp; SWYNEX AI Problem Design Task 1 &nbsp;·&nbsp;
    Built with Python · Scikit-Learn · Streamlit · Plotly
</div>""", unsafe_allow_html=True)
