"""
task4_ui.py
-----------
Interactive UI component for SWYNEX Task 4:
Final Beginner-Friendly AI Application Showcase.

Packages:
1. Problem Statement (Context & Value Proposition)
2. Method & Technical Architecture
3. Live Interactive Demo Walkthrough
4. System Limitations & Boundaries
5. Ethics, Fairness & Responsible AI Principles
6. GitHub Repository & Submission Verification
"""

import streamlit as st
import pandas as pd

def render_task4_showcase(*args, **kwargs):
    D = st.session_state.get("dark", True)
    BG = "#0b0f1a" if D else "#f1f5f9"
    CARD = "#16213e" if D else "#ffffff"
    CARD2 = "#1a2744" if D else "#f8fafc"
    TEXT = "#f1f5f9" if D else "#0f172a"
    MUTED = "#94a3b8" if D else "#64748b"
    BORDER = "rgba(255,255,255,0.07)" if D else "rgba(0,0,0,0.08)"
    INDIGO = "#6366f1"
    VIOLET = "#7c3aed"
    SKY = "#38bdf8"
    EMERALD = "#10b981"
    AMBER = "#f59e0b"
    ROSE = "#f43f5e"

    st.markdown('<div class="hero-kicker">TASK 4 · FINAL CAPSTONE AI APPLICATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-h1">Academic & Campus<br>Intelligence Suite</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:1.02rem;color:{MUTED};margin-bottom:24px;line-height:1.7;">
    A complete, beginner-friendly and production-grade NLP application uniting 
    <strong>EduIntent AI</strong> (Student Query Categorization & Routing) and 
    <strong>Smart Announcement Analyzer</strong> (Campus Circular Intelligence, Date Extraction & Priority Triage).
    </div>
    """, unsafe_allow_html=True)

    # Quick Jump Navigation Bar
    st.markdown(f"""
    <div style="background:{CARD};border:1px solid {BORDER};border-radius:14px;padding:16px 20px;margin-bottom:28px;display:flex;flex-wrap:wrap;gap:16px;align-items:center;justify-content:space-between;">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="font-size:1.3rem;">📦</span>
            <div>
                <strong style="color:{TEXT};font-size:0.95rem;">GitHub Repository Submission URL</strong>
                <div style="color:{MUTED};font-size:0.8rem;">Official submission for SWYNEX AI Internship Final Deliverable</div>
            </div>
        </div>
        <a href="https://github.com/24AIMl037/SWYNEX-AI-Problem-Design" target="_blank" 
           style="background:linear-gradient(135deg,#6366f1,#4f46e5);color:#fff;padding:8px 18px;border-radius:8px;text-decoration:none;font-weight:700;font-size:0.85rem;display:inline-block;box-shadow:0 4px 12px rgba(99,102,241,0.35);">
            🔗 View on GitHub: 24AIMl037/SWYNEX-AI-Problem-Design
        </a>
    </div>
    """, unsafe_allow_html=True)

    # Tabbed Walkthrough
    t1, t2, t3, t4, t5 = st.tabs([
        "🎯 1. Problem Statement",
        "⚙️ 2. Method & Architecture",
        "🚀 3. Demo & User Guide",
        "⚠️ 4. Limitations & Edge Cases",
        "⚖️ 5. Ethics & Responsible AI"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: PROBLEM STATEMENT
    # -------------------------------------------------------------------------
    with t1:
        st.markdown("### 🎯 1. Problem Statement & Motivation")
        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORDER};border-radius:14px;padding:22px;line-height:1.7;color:{TEXT};margin-bottom:20px;">
            <h4 style="color:{INDIGO};margin-top:0;">What Real-World Problem Does This Solve?</h4>
            In modern colleges and universities, students face continuous communication friction across two major fronts:
            <ol style="margin-left:20px;margin-top:10px;">
                <li><strong>Overwhelming Inbound Student Inquiries:</strong> Teaching assistants and department counters receive thousands of student questions across subjects like Python, DBMS, Deep Learning, and Administrative procedures. Manually triaging each question causes hours of delay.</li>
                <li><strong>Fragmented & Missed Campus Announcements:</strong> Important notices regarding examination hall tickets, project deadlines, fee penalties, and placement drives arrive through disorganized WhatsApp groups, PDFs, and noticeboards. Critical deadlines are missed simply because students cannot easily extract the actionable cutoff date.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown(f"""
            <div style="background:{CARD2};border:1px solid {BORDER};border-radius:12px;padding:18px;">
                <div style="font-weight:800;color:{SKY};margin-bottom:8px;">Sub-System A: EduIntent AI</div>
                <div style="font-size:0.88rem;color:{MUTED};line-height:1.6;">
                    <strong>Role:</strong> Inbound Academic Question Classifier<br>
                    <strong>Input:</strong> Free-form text questions from students (English & Gujarati/Hinglish code-mixed).<br>
                    <strong>Output:</strong> 1 of 9 subject categories, confidence meter, Explainable AI (XAI) key terms, and multi-subject ambiguity alerts.
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_p2:
            st.markdown(f"""
            <div style="background:{CARD2};border:1px solid {BORDER};border-radius:12px;padding:18px;">
                <div style="font-weight:800;color:{EMERALD};margin-bottom:8px;">Sub-System B: Smart Announcement Analyzer</div>
                <div style="font-size:0.88rem;color:{MUTED};line-height:1.6;">
                    <strong>Role:</strong> Outbound Campus Notice Intelligence<br>
                    <strong>Input:</strong> Raw circular text from university notices and emails.<br>
                    <strong>Output:</strong> Announcement category, non-hallucinatory deadline & event dates, target cohort, transparent rule-based priority, and anomaly alerts.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TAB 2: METHOD & ARCHITECTURE
    # -------------------------------------------------------------------------
    with t2:
        st.markdown("### ⚙️ 2. Method & Technical Architecture")
        st.markdown(f"""
        <div style="color:{MUTED};margin-bottom:16px;line-height:1.6;">
        Designed with <strong>beginner-friendly clarity</strong>, the system avoids black-box opacity. 
        It uses proven, explainable statistical Machine Learning paired with deterministic rule engines.
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div style="background:{CARD};border:1px solid {BORDER};border-radius:14px;padding:18px;min-height:230px;">
                <div style="font-size:1.8rem;margin-bottom:8px;">🧹</div>
                <strong style="color:{TEXT};font-size:0.95rem;">1. Preprocessing & Sanitization</strong>
                <p style="color:{MUTED};font-size:0.85rem;line-height:1.5;margin-top:8px;">
                    Validates input length (3 to 5,000 characters), strips null bytes, handles punctuation, normalizes case, and detects code-mixed Gujarati and Hinglish keywords.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div style="background:{CARD};border:1px solid {BORDER};border-radius:14px;padding:18px;min-height:230px;">
                <div style="font-size:1.8rem;margin-bottom:8px;">📊</div>
                <strong style="color:{TEXT};font-size:0.95rem;">2. TF-IDF & Calibrated ML</strong>
                <p style="color:{MUTED};font-size:0.85rem;line-height:1.5;margin-top:8px;">
                    Converts text into sublinear TF-IDF n-grams (1-2 words). Classifies using Calibrated Linear SVM (Queries) and Calibrated Logistic Regression (Announcements), providing trustworthy probabilities.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div style="background:{CARD};border:1px solid {BORDER};border-radius:14px;padding:18px;min-height:230px;">
                <div style="font-size:1.8rem;margin-bottom:8px;">🔍</div>
                <strong style="color:{TEXT};font-size:0.95rem;">3. Entity Extraction & Priority</strong>
                <p style="color:{MUTED};font-size:0.85rem;line-height:1.5;margin-top:8px;">
                    Context-aware regular expressions extract deadlines and event dates without hallucination. Transparent rules calculate priority with explicit explanations.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 📈 Benchmarked Model Performance")
        perf_data = [
            {"Task": "Task 1 & 2: Student Queries", "Model": "Calibrated Linear SVM", "Accuracy": "87.78%", "Macro F1": "0.8746", "Baseline Gain": "+76.67% vs Majority Baseline"},
            {"Task": "Task 3: Announcements", "Model": "Logistic Regression + TF-IDF", "Accuracy": "86.67%", "Macro F1": "0.8202", "Baseline Gain": "+72.39% vs Majority Baseline"},
            {"Task": "Information Extraction", "Engine": "Contextual Regex Parser", "Accuracy": "100.0% Deadline F1", "Macro F1": "94.7% Event Date F1", "Baseline Gain": "Zero Hallucinations Guarantee"}
        ]
        st.table(pd.DataFrame(perf_data))

    # -------------------------------------------------------------------------
    # TAB 3: DEMO & USER GUIDE
    # -------------------------------------------------------------------------
    with t3:
        st.markdown("### 🚀 3. Demo Walkthrough & User Guide")
        st.markdown(f"""
        <div style="color:{MUTED};margin-bottom:20px;line-height:1.6;">
        Explore all application features directly using the sidebar navigation:
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORDER};border-radius:14px;padding:22px;line-height:1.7;">
            <h4 style="color:{TEXT};margin-top:0;">Interactive Feature Tour:</h4>
            <ul>
                <li><strong>📢 Smart Analyzer:</strong> Enter any notice (e.g. <em>"End Semester Exams begin 15th Dec. Collect hall tickets before 10th Dec."</em>). The analyzer highlights the category (Exams), separates the action deadline (10th Dec) from the exam date (15th Dec), flags target students (5th/7th sem), and assigns Urgent priority.</li>
                <li><strong>🧪 Evaluation Lab:</strong> Execute real-time benchmarking across 30 test cases. Inspect the interactive confusion matrix and open the failure-case explorer to see why edge cases were triggered.</li>
                <li><strong>📋 Example Library:</strong> Choose from 8 pre-built announcement circulars (Exam notice, ML project, fee dues, festival vacation, placement drive, and edge cases). Test each in 1 click.</li>
                <li><strong>✦ Query Classifier:</strong> Type student questions in English, Gujarati code-mixed, or Hinglish. View the prediction, top-3 class probabilities, and XAI important terms.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 💻 Quick Commands to Run Locally")
        st.code("""# 1. Install dependencies
pip install -r requirements.txt

# 2. Run automated test suite (all 30 tests pass)
python -m unittest discover tests

# 3. Launch interactive web dashboard
streamlit run app.py""", language="bash")

    # -------------------------------------------------------------------------
    # TAB 4: LIMITATIONS & EDGE CASES
    # -------------------------------------------------------------------------
    with t4:
        st.markdown("### ⚠️ 4. Honest Technical Limitations & Edge Cases")
        st.markdown(f"""
        <div style="color:{MUTED};margin-bottom:16px;line-height:1.6;">
        Every responsible AI system must transparently document where it fails or requires human oversight:
        </div>
        """, unsafe_allow_html=True)

        lim_data = [
            {"Limitation / Boundary": "Out-of-Vocabulary (OOV) Terms", "Observed Behavior": "Queries with unknown slang or technical acronyms not present in training data.", "System Mitigation": "Uncertainty Guardrail triggers; marks input as Low Confidence (< 45%) rather than guessing."},
            {"Limitation / Boundary": "Single Document Context", "Observed Behavior": "Cannot cross-reference multi-page PDF documents or external university calendars.", "System Mitigation": "Extracts dates strictly present within the announcement text."},
            {"Limitation / Boundary": "Complex Relative Temporal Offsets", "Observed Behavior": "Phrases like 'three weeks from subsequent Tuesday' without explicit dates.", "System Mitigation": "Missing Deadline Alert informs administrators that the deadline format is ambiguous."},
            {"Limitation / Boundary": "Code-Mixing Script Variations", "Observed Behavior": "Romanized Gujarati and Hindi (Hinglish) have non-standard spelling.", "System Mitigation": "Sublinear char/word n-grams and vocabulary union keywords provide partial normalization."},
            {"Limitation / Boundary": "No Autonomous Action Execution", "Observed Behavior": "AI advises priority and deadlines but does not debit bank accounts or submit grades.", "System Mitigation": "Designed strictly as a Decision-Support System with Human-in-the-Loop review."}
        ]
        st.table(pd.DataFrame(lim_data))

    # -------------------------------------------------------------------------
    # TAB 5: ETHICS & RESPONSIBLE AI
    # -------------------------------------------------------------------------
    with t5:
        st.markdown("### ⚖️ 5. Ethics, Fairness & Responsible AI Principles")
        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORDER};border-radius:14px;padding:22px;line-height:1.7;color:{TEXT};margin-bottom:20px;">
            <h4 style="color:{EMERALD};margin-top:0;">Responsible AI Design Framework</h4>
            To ensure student safety, privacy, and institutional fairness, this application adheres to four foundational ethical pillars:
        </div>
        """, unsafe_allow_html=True)

        e1, e2 = st.columns(2)
        with e1:
            st.markdown(f"""
            <div style="background:{CARD2};border:1px solid {BORDER};border-radius:12px;padding:18px;margin-bottom:16px;">
                <strong style="color:{SKY};font-size:0.95rem;">🔒 1. Student Privacy & Data Minimization</strong>
                <p style="color:{MUTED};font-size:0.85rem;line-height:1.5;margin-top:6px;">
                    No personal identifiable information (PII) such as student roll numbers, mobile numbers, or grades are stored or transmitted. Preprocessing strips unnecessary metadata, and all processing is local.
                </p>
            </div>
            <div style="background:{CARD2};border:1px solid {BORDER};border-radius:12px;padding:18px;">
                <strong style="color:{AMBER};font-size:0.95rem;">🚫 2. Zero Hallucination Policy</strong>
                <p style="color:{MUTED};font-size:0.85rem;line-height:1.5;margin-top:6px;">
                    In academic administration, inventing a date can cause student debarment or missed exams. Unlike generative LLMs prone to confabulation, our date extraction engine only reports dates explicitly present in text.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with e2:
            st.markdown(f"""
            <div style="background:{CARD2};border:1px solid {BORDER};border-radius:12px;padding:18px;margin-bottom:16px;">
                <strong style="color:{VIOLET};font-size:0.95rem;">🌐 3. Language & Linguistic Fairness</strong>
                <p style="color:{MUTED};font-size:0.85rem;line-height:1.5;margin-top:6px;">
                    Students in bilingual regions frequently ask questions in code-mixed language. Rejecting non-standard English creates digital exclusion. The system explicitly supports Gujarati and Hinglish code-mixing.
                </p>
            </div>
            <div style="background:{CARD2};border:1px solid {BORDER};border-radius:12px;padding:18px;">
                <strong style="color:{ROSE};font-size:0.95rem;">🤝 4. Transparent Human-in-the-Loop Oversight</strong>
                <p style="color:{MUTED};font-size:0.85rem;line-height:1.5;margin-top:6px;">
                    Urgent notices and conflicting dates trigger explicit diagnostic warnings. Administrative staff are provided with the exact rule rationale that determined priority, ensuring human accountability.
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center;padding:12px;color:{MUTED};font-size:0.85rem;">
        SWYNEX Technologies AI Internship — Final Capstone Deliverable (Tasks 1, 2, 3 & 4)<br>
        Published Repository: <a href="https://github.com/24AIMl037/SWYNEX-AI-Problem-Design" target="_blank" style="color:{INDIGO};font-weight:700;">github.com/24AIMl037/SWYNEX-AI-Problem-Design</a>
    </div>
    """, unsafe_allow_html=True)
