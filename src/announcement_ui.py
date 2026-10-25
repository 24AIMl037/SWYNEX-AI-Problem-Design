"""
announcement_ui.py
------------------
Modular Streamlit UI components for SWYNEX Task 3:
Student Announcement Intelligence System.

Provides:
1. render_smart_analyzer() - Smart Announcement Analyzer UI
2. render_evaluation_lab() - Evaluation Lab with genuine live metrics and failure explorer
3. render_example_library() - Catalog of test cases with 1-click test actions
4. render_task3_documentation() - Technical specifications and rule engine matrix
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any

# Theme Palette Defaults
BG = "#0b0f1a"
CARD = "#16213e"
CARD2 = "#1a2744"
TEXT = "#f1f5f9"
MUTED = "#94a3b8"
BORDER = "rgba(255,255,255,0.07)"
INDIGO = "#6366f1"
VIOLET = "#7c3aed"
SKY = "#38bdf8"
EMERALD = "#10b981"
AMBER = "#f59e0b"
ROSE = "#f43f5e"

CATEGORY_COLORS = {
    "Exams": "#8b5cf6",       # Violet
    "Assignments": "#3b82f6", # Blue
    "Events": "#ec4899",      # Pink
    "Holidays": "#f59e0b",    # Amber
    "Placements": "#10b981",  # Emerald
    "Fees": "#f43f5e",        # Rose
    "General": "#06b6d4",     # Cyan
    "Unsupported / Low Confidence": "#64748b" # Muted Slate
}

CATEGORY_ICONS = {
    "Exams": "📝",
    "Assignments": "📚",
    "Events": "🎉",
    "Holidays": "🏖️",
    "Placements": "💼",
    "Fees": "💳",
    "General": "🏛️",
    "Unsupported / Low Confidence": "🛑"
}

PRIORITY_CONFIG = {
    "Urgent / High": {
        "color": "#f43f5e",
        "bg": "rgba(244, 63, 94, 0.15)",
        "border": "#f43f5e",
        "icon": "🚨"
    },
    "Medium": {
        "color": "#f59e0b",
        "bg": "rgba(245, 158, 11, 0.15)",
        "border": "#f59e0b",
        "icon": "⚡"
    },
    "Low / Informational": {
        "color": "#38bdf8",
        "bg": "rgba(56, 189, 248, 0.15)",
        "border": "#38bdf8",
        "icon": "ℹ️"
    }
}

PRESET_ANNOUNCEMENTS = [
    {
        "label": "📝 [Exams] End-Semester Theory Exams & Hall Tickets",
        "text": "End Semester Theory Examinations for 5th and 7th semester B.Tech will commence from 15th December 2026. Hall tickets must be collected before 10th December 2026."
    },
    {
        "label": "📚 [Assignments] Machine Learning Assignment 3 & Mini-Project",
        "text": "All 3rd year Computer Engineering students must submit Assignment 3 for Machine Learning on Google Classroom by 12th November 2026 before 11:59 PM."
    },
    {
        "label": "💳 [Fees] Semester Tuition Fees with Late Fine Penalty",
        "text": "The last date for payment of Even Semester Tuition Fees is 30th November 2026. A late fine of Rs. 100 per day will be charged strictly after the due date."
    },
    {
        "label": "🎉 [Events] Technovation 2026 National Technical Fest",
        "text": "Annual National Technical Symposium 'TECHNOVATION 2026' will be organized on 18th December 2026. All students must register before 5th December 2026 on the fest portal."
    },
    {
        "label": "🏖️ [Holidays] Diwali Vacation & Academic Recess",
        "text": "The college and administrative offices will remain closed from 11th November 2026 to 18th November 2026 on the auspicious occasion of Diwali festival. Classes resume on 19th November."
    },
    {
        "label": "💼 [Placements] Microsoft India Campus Recruitment Drive",
        "text": "Microsoft India campus placement drive for final year B.Tech CSE and IT students. Eligible candidates must upload their updated resume on the T&P portal before 15th November 2026."
    },
    {
        "label": "🏛️ [General] Campus ID Card Requirement & Security Check",
        "text": "All students must wear their official college identity cards around their neck while inside campus premises. Security will strictly check ID cards at Gate 1."
    },
    {
        "label": "🔀 [Ambiguous] Supplementary Re-Exam Form with Examination Fee",
        "text": "Students with backlogs must pay the supplementary examination fee of Rs. 500 and submit the re-exam registration form before 18th November 2026."
    },
    {
        "label": "⚠️ [Missing Deadline] Urgent Mess Dues Clearance (No Date)",
        "text": "All hostel students are urgently required to clear their outstanding mess fee arrears immediately. Failure to do so will result in room cancellation."
    },
    {
        "label": "❌ [Conflicting Dates] Event Date Scheduled Before Registration",
        "text": "The Annual Tech Hackathon 'CodeStorm' will be held on 10th November 2026. However, team registrations on the event portal will open on 20th November 2026."
    },
    {
        "label": "🛑 [Out-of-Domain] Chocolate Fudge Cake Recipe",
        "text": "To bake a classic chocolate fudge cake, preheat the oven to 350 degrees Fahrenheit. Mix all-purpose flour, granulated sugar, dark cocoa powder, baking soda, and eggs."
    }
]

# ═══════════════════════════════════════════════════════════════════════════════
# 1. SMART ANNOUNCEMENT ANALYZER UI
# ═══════════════════════════════════════════════════════════════════════════════
def render_smart_analyzer(analyzer, *args, **kwargs):
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

    st.markdown('<div class="hero-kicker">TASK 3 INTELLIGENT FEATURE · SWYNEX AI INTERNSHIP</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-h1">Smart Announcement<br>Analyzer</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.95rem;color:{MUTED};margin-bottom:24px;line-height:1.6;">
    An advanced Natural Language Processing system that automatically categorizes college announcements,
    extracts actionable deadlines, event dates, and target audiences, estimates priority with transparent rule rationale,
    and flags ambiguities, missing deadlines, or conflicting dates through verification guardrails.
    </div>
    """, unsafe_allow_html=True)

    # Preset Selector Row
    c_preset, c_btn = st.columns([3.5, 1])
    with c_preset:
        preset_options = ["-- Select a Pre-Loaded College Announcement --"] + [p["label"] for p in PRESET_ANNOUNCEMENTS]
        selected_preset = st.selectbox(
            "📋 Quick Sample Announcements",
            preset_options,
            label_visibility="collapsed"
        )
    with c_btn:
        if st.button("🔄 Clear Input", use_container_width=True):
            st.session_state["announcement_input"] = ""
            st.session_state["announcement_result"] = None
            st.rerun()

    # Load text from preset if chosen
    if selected_preset != "-- Select a Pre-Loaded College Announcement --":
        for p in PRESET_ANNOUNCEMENTS:
            if p["label"] == selected_preset:
                st.session_state["announcement_input"] = p["text"]
                break

    current_text = st.session_state.get("announcement_input", "")
    
    # Input Text Area
    announcement_text = st.text_area(
        "Enter or paste college announcement circular:",
        value=current_text,
        height=140,
        placeholder="e.g., Final Year B.Tech students must submit their project report and clear library dues before 15th December 2026. Late submissions will incur Rs. 200 fine.",
        key="ann_input_area"
    )
    st.session_state["announcement_input"] = announcement_text

    col_meta1, col_meta2 = st.columns([2, 1])
    with col_meta1:
        words = len(announcement_text.split()) if announcement_text else 0
        chars = len(announcement_text) if announcement_text else 0
        st.caption(f"📊 Length: **{chars}** characters | **{words}** words")
    with col_meta2:
        analyze_clicked = st.button("🚀 Analyze Announcement", type="primary", use_container_width=True)

    if analyze_clicked or (announcement_text and "announcement_result" not in st.session_state):
        if not announcement_text.strip():
            st.warning("⚠️ Please enter announcement text to analyze.")
        else:
            with st.spinner("Analyzing semantics, extracting entities & applying priority rules..."):
                res = analyzer.analyze(announcement_text)
                st.session_state["announcement_result"] = res

    # Render Results
    res = st.session_state.get("announcement_result")
    if res and res.get("raw_text"):
        st.markdown("---")
        st.markdown("### 🎯 Intelligence Analysis Results")

        cat = res["predicted_category"]
        conf = res["confidence"]
        priority = res["priority"]
        cat_color = CATEGORY_COLORS.get(cat, INDIGO)
        cat_icon = CATEGORY_ICONS.get(cat, "📢")
        p_cfg = PRIORITY_CONFIG.get(priority, PRIORITY_CONFIG["Medium"])

        # Top Summary Card
        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORDER};border-radius:16px;padding:22px;margin-bottom:20px;
            box-shadow:0 8px 24px rgba(0,0,0,0.15);">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px;">
                <div>
                    <div style="font-size:0.75rem;font-weight:700;color:{MUTED};text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">
                        PREDICTED CATEGORY
                    </div>
                    <div style="display:flex;align-items:center;gap:10px;">
                        <span style="font-size:2rem;">{cat_icon}</span>
                        <span style="font-size:1.6rem;font-weight:900;color:{cat_color};">{cat}</span>
                        <span style="background:{cat_color}22;color:{cat_color};border:1px solid {cat_color}44;
                            padding:3px 10px;border-radius:20px;font-size:0.85rem;font-weight:700;">
                            {conf*100:.1f}% Confidence
                        </span>
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:0.75rem;font-weight:700;color:{MUTED};text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">
                        PRIORITY LEVEL
                    </div>
                    <div style="display:inline-flex;align-items:center;gap:8px;background:{p_cfg['bg']};
                        border:1px solid {p_cfg['border']};padding:8px 16px;border-radius:12px;">
                        <span style="font-size:1.2rem;">{p_cfg['icon']}</span>
                        <span style="font-size:1.15rem;font-weight:900;color:{p_cfg['color']};">{priority}</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Priority Rule Rationale Banner
        with st.expander("🔍 Priority Estimation Rule Rationale (Why this priority?)", expanded=True):
            st.markdown(f"**Calculated Priority:** `{priority}`")
            st.markdown("Rules triggered by the priority estimation engine:")
            for r in res["priority_rationales"]:
                st.markdown(f"- ✅ **{r}**")

        # 4-Column Extracted Entities Grid
        st.markdown("#### 📌 Extracted Information & Entities")
        e1, e2, e3, e4 = st.columns(4)

        with e1:
            dl_content = ""
            if res["deadlines"]:
                for d in res["deadlines"]:
                    dl_content += f'<div style="background:{ROSE}22;color:{ROSE};border:1px solid {ROSE}44;padding:6px 10px;border-radius:8px;font-weight:700;margin-bottom:6px;font-size:0.85rem;">⏳ {d}</div>'
            else:
                dl_content = f'<div style="color:{MUTED};font-size:0.85rem;font-style:italic;">No deadline detected</div>'

            st.markdown(f"""
            <div style="background:{CARD2};border:1px solid {BORDER};border-radius:12px;padding:16px;min-height:140px;">
                <div style="font-size:0.75rem;font-weight:700;color:{ROSE};text-transform:uppercase;margin-bottom:8px;">
                    📅 DEADLINES & CUTOFFS
                </div>
                {dl_content}
            </div>
            """, unsafe_allow_html=True)

        with e2:
            ev_content = ""
            if res["event_dates"]:
                for ev in res["event_dates"]:
                    ev_content += f'<div style="background:{SKY}22;color:{SKY};border:1px solid {SKY}44;padding:6px 10px;border-radius:8px;font-weight:700;margin-bottom:6px;font-size:0.85rem;">🗓️ {ev}</div>'
            else:
                ev_content = f'<div style="color:{MUTED};font-size:0.85rem;font-style:italic;">No event dates detected</div>'

            st.markdown(f"""
            <div style="background:{CARD2};border:1px solid {BORDER};border-radius:12px;padding:16px;min-height:140px;">
                <div style="font-size:0.75rem;font-weight:700;color:{SKY};text-transform:uppercase;margin-bottom:8px;">
                    🗓️ EVENT / COMMENCEMENT
                </div>
                {ev_content}
            </div>
            """, unsafe_allow_html=True)

        with e3:
            aud = res["target_audience"]
            if aud:
                aud_content = f'<div style="background:{VIOLET}22;color:#a78bfa;border:1px solid {VIOLET}44;padding:6px 10px;border-radius:8px;font-weight:700;font-size:0.85rem;">👥 {aud}</div>'
            else:
                aud_content = f'<div style="color:{MUTED};font-size:0.85rem;font-style:italic;">General campus audience</div>'

            st.markdown(f"""
            <div style="background:{CARD2};border:1px solid {BORDER};border-radius:12px;padding:16px;min-height:140px;">
                <div style="font-size:0.75rem;font-weight:700;color:#a78bfa;text-transform:uppercase;margin-bottom:8px;">
                    👥 TARGET AUDIENCE
                </div>
                {aud_content}
            </div>
            """, unsafe_allow_html=True)

        with e4:
            act = res["required_action"]
            if act:
                act_content = f'<div style="background:{EMERALD}22;color:{EMERALD};border:1px solid {EMERALD}44;padding:6px 10px;border-radius:8px;font-weight:700;font-size:0.85rem;">⚡ {act}</div>'
            else:
                act_content = f'<div style="color:{MUTED};font-size:0.85rem;font-style:italic;">Informational notice</div>'

            st.markdown(f"""
            <div style="background:{CARD2};border:1px solid {BORDER};border-radius:12px;padding:16px;min-height:140px;">
                <div style="font-size:0.75rem;font-weight:700;color:{EMERALD};text-transform:uppercase;margin-bottom:8px;">
                    ⚡ REQUIRED ACTION
                </div>
                {act_content}
            </div>
            """, unsafe_allow_html=True)

        # Verification Warnings & Anomaly Alerts
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        if res["has_warnings"]:
            st.markdown("#### ⚠️ Verification Warnings & Guardrail Alerts")
            for w in res["warnings"]:
                if "missing" in w.lower():
                    st.warning(f"⚠️ **Missing Deadline Alert:** {w}")
                elif "conflict" in w.lower():
                    st.error(f"❌ **Conflicting Dates Alert:** {w}")
                elif "ambiguous" in w.lower():
                    st.info(f"🔀 **Multi-Category Overlap:** {w}")
                elif "unsupported" in w.lower() or "low prediction" in w.lower():
                    st.error(f"🛑 **Verification Guardrail:** {w}")
                else:
                    st.warning(f"⚠️ {w}")
        else:
            st.success("✅ **Verification Passed:** Prediction confidence is robust and no chronological or deadline anomalies were detected.")

        # Class Probability Distribution Chart
        if res["top_categories"]:
            st.markdown("#### 📊 Candidate Category Probabilities")
            cats = [c["category"] for c in res["top_categories"]]
            probs = [c["probability"] * 100 for c in res["top_categories"]]
            colors = [CATEGORY_COLORS.get(c, INDIGO) for c in cats]

            fig = go.Figure(go.Bar(
                x=probs,
                y=cats,
                orientation="h",
                marker=dict(color=colors),
                text=[f"{p:.1f}%" for p in probs],
                textposition="auto",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT, family="Inter"),
                height=190,
                margin=dict(l=20, r=20, t=10, b=20),
                xaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.05)" if D else "#e2e8f0"),
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. EVALUATION LAB UI
# ═══════════════════════════════════════════════════════════════════════════════
def render_evaluation_lab(lab, *args, **kwargs):
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

    st.markdown('<div class="hero-kicker">TASK 3 VERIFICATION & TESTING · EVALUATION LAB</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-h1">Evaluation Lab</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.95rem;color:{MUTED};margin-bottom:24px;line-height:1.6;">
    An interactive testing framework that benchmarks the system against 30 manually annotated announcements,
    including normal notices, ambiguous queries, conflicting dates, missing deadlines, out-of-domain text, and malformed inputs.
    Every metric displayed is measured live; no scores are fabricated.
    </div>
    """, unsafe_allow_html=True)

    c_run, c_meta = st.columns([1.5, 3])
    with c_run:
        run_btn = st.button("▶️ Execute Benchmark Suite", type="primary", use_container_width=True)
    with c_meta:
        st.caption(f"📁 Benchmark dataset: `dataset/evaluation_announcements.json` ({len(lab.test_cases)} annotated test cases)")

    if run_btn or "eval_results" not in st.session_state:
        with st.spinner("Running live evaluation on 30 test cases across 6 typologies..."):
            results = lab.run_evaluation()
            st.session_state["eval_results"] = results

    results = st.session_state.get("eval_results")
    if not results or "error" in results:
        st.error(results.get("error", "Failed to load evaluation results."))
        return

    c = results["classification"]
    e = results["extraction"]
    tb = results["type_breakdown"]
    failures = results["failure_cases"]

    # Headline KPI Cards
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpis = [
        (f"{c['accuracy']*100:.1f}%", "Overall Accuracy", "30 test cases"),
        (f"{c['f1_macro']:.4f}", "Macro F1-Score", "Balanced across classes"),
        (f"{e['deadlines']['f1']*100:.1f}%", "Deadline F1", f"P: {e['deadlines']['precision']*100:.0f}% | R: {e['deadlines']['recall']*100:.0f}%"),
        (f"{e['event_dates']['f1']*100:.1f}%", "Event Date F1", f"P: {e['event_dates']['precision']*100:.0f}% | R: {e['event_dates']['recall']*100:.0f}%"),
        (f"{e['actions']['f1']*100:.1f}%", "Action F1", f"P: {e['actions']['precision']*100:.0f}% | R: {e['actions']['recall']*100:.0f}%"),
        (f"{e['guardrail_detection_rate']*100:.1f}%", "Guardrail Recall", "Anomalies flagged"),
    ]
    for col, (val, label, sub) in zip([k1, k2, k3, k4, k5, k6], kpis):
        with col:
            st.markdown(f"""
            <div style="background:{CARD};border:1px solid {BORDER};border-radius:14px;padding:16px;text-align:center;">
                <div style="font-size:1.65rem;font-weight:900;color:{TEXT};">{val}</div>
                <div style="font-size:0.75rem;font-weight:700;color:{INDIGO};text-transform:uppercase;margin:4px 0;">{label}</div>
                <div style="font-size:0.75rem;color:{MUTED};">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Typology Breakdown & Confusion Matrix
    col_chart1, col_chart2 = st.columns([1.2, 1.8])

    with col_chart1:
        st.markdown("#### 📈 Performance by Typology")
        types = list(tb.keys())
        accs = [tb[t]["accuracy"] * 100 for t in types]
        type_labels = [t.replace("_", " ").title() for t in types]

        fig_typ = go.Figure(go.Bar(
            x=accs,
            y=type_labels,
            orientation="h",
            marker=dict(color=INDIGO),
            text=[f"{a:.1f}%" for a in accs],
            textposition="auto"
        ))
        fig_typ.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT, family="Inter"),
            height=300,
            margin=dict(l=10, r=20, t=10, b=20),
            xaxis=dict(range=[0, 110], showgrid=True, gridcolor="rgba(255,255,255,0.05)" if D else "#e2e8f0"),
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_typ, use_container_width=True)

    with col_chart2:
        st.markdown("#### 🎯 Confusion Matrix")
        cm_df = pd.DataFrame(c["confusion_matrix"], index=c["classes"], columns=c["classes"])
        fig_cm = px.imshow(
            cm_df,
            labels=dict(x="Predicted Class", y="Ground Truth Class", color="Count"),
            x=c["classes"],
            y=c["classes"],
            color_continuous_scale="Blues",
            text_auto=True
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT, family="Inter"),
            height=300,
            margin=dict(l=10, r=10, t=10, b=20)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # Detailed Test Case Explorer
    st.markdown("---")
    st.markdown(f"#### 🧪 Live Test Case Results ({results['total_tests']} Total | {results['passed_tests']} Correct | {len(failures)} Failure Cases)")

    tab_all, tab_failures = st.tabs(["📋 All Executed Test Cases", f"⚠️ Recorded Failure Cases ({len(failures)})"])

    with tab_all:
        df_all = pd.DataFrame([
            {
                "ID": r["id"],
                "Typology": r["test_type"].replace("_", " ").title(),
                "Text": r["text"],
                "Expected": r["expected_category"],
                "Predicted": r["predicted_category"],
                "Match": "✅ Correct" if r["is_category_correct"] else "❌ Mismatch",
                "Confidence": f"{r['confidence']*100:.1f}%",
                "Priority": r["priority"]
            }
            for r in results["detailed_results"]
        ])
        st.dataframe(df_all, use_container_width=True, hide_index=True)

    with tab_failures:
        if not failures:
            st.success("🎉 No failure cases recorded!")
        else:
            for idx, f in enumerate(failures, 1):
                with st.expander(f"Case #{idx}: [{f['id']}] {f['test_type'].upper()} — {f['text'][:65]}...", expanded=False):
                    col_f1, col_f2 = st.columns(2)
                    with col_f1:
                        st.markdown(f"**Expected Category:** `{f['expected_category']}`")
                        st.markdown(f"**Actual Predicted:** `{f['predicted_category']}`")
                        st.markdown(f"**Discrepancies:** {', '.join(f['failure_reasons'])}")
                    with col_f2:
                        st.markdown(f"**Root Cause:** {f['root_cause']}")
                        st.markdown(f"**Suggested Improvement:** {f['suggested_improvement']}")

    # Report Download Actions
    st.markdown("---")
    cd1, cd2 = st.columns(2)
    with cd1:
        md_report = lab.generate_markdown_report(results)
        st.download_button(
            "📥 Download Evaluation Report (.md)",
            data=md_report,
            file_name="evaluation_report.md",
            mime="text/markdown",
            use_container_width=True
        )
    with cd2:
        json_report = json.dumps(results, indent=2)
        st.download_button(
            "📥 Export Raw Results (.json)",
            data=json_report,
            file_name="evaluation_results.json",
            mime="application/json",
            use_container_width=True
        )

# ═══════════════════════════════════════════════════════════════════════════════
# 3. EXAMPLE ANNOUNCEMENTS LIBRARY UI
# ═══════════════════════════════════════════════════════════════════════════════
def render_example_library(*args, **kwargs):
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

    st.markdown('<div class="hero-kicker">PRESET TEST SAMPLES · TASK 3</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-h1">Example Library</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.95rem;color:{MUTED};margin-bottom:24px;line-height:1.6;">
    Curated library of realistic college announcement circulars illustrating normal academic notices,
    ambiguous overlaps, missing deadlines, conflicting dates, and out-of-domain edge cases.
    Click <strong>Test in Analyzer</strong> on any card to run live inference immediately.
    </div>
    """, unsafe_allow_html=True)

    for idx, sample in enumerate(PRESET_ANNOUNCEMENTS):
        col_c, col_act = st.columns([4, 1.2])
        with col_c:
            st.markdown(f"""
            <div style="background:{CARD};border:1px solid {BORDER};border-radius:12px;padding:16px;margin-bottom:12px;">
                <div style="font-weight:700;color:{TEXT};font-size:0.95rem;margin-bottom:6px;">{sample['label']}</div>
                <div style="color:{MUTED};font-size:0.88rem;line-height:1.5;">{sample['text']}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_act:
            if st.button("🚀 Test in Analyzer", key=f"btn_test_{idx}", use_container_width=True):
                st.session_state["announcement_input"] = sample["text"]
                st.session_state["active_nav"] = "📢 Smart Analyzer"
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# 4. TASK 3 DOCUMENTATION & RULES UI
# ═══════════════════════════════════════════════════════════════════════════════
def render_task3_documentation(*args, **kwargs):
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

    st.markdown('<div class="hero-kicker">TECHNICAL SPECIFICATIONS & ARCHITECTURE · TASK 3</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-h1">Documentation & Rules</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🏛️ System Architecture Overview
    The **Student Announcement Intelligence System** is architected as an end-to-end NLP intelligence engine:
    1. **Preprocessing & Sanitization Layer (`validation.py`):**
       - Strips null characters, enforces minimum (3) and maximum (5,000) length boundaries.
       - Rejects symbols-only or empty strings with explanatory diagnostic feedback.
    2. **Machine Learning Classifier (`announcement_model.py`):**
       - **Feature Extraction:** Sublinear TF-IDF vectorizer (ngram 1-2, 4,000 max features, English stop words removal).
       - **Classification:** Scikit-Learn Logistic Regression with calibrated class probabilities.
       - **Classes (7):** `Exams`, `Assignments`, `Events`, `Holidays`, `Placements`, `Fees`, `General`.
    3. **Deterministic Information Extraction Engine (`announcement_analyzer.py`):**
       - **Deadlines:** Contextual regex parsing of submission/payment cutoffs.
       - **Event Dates:** Event scheduling, commencement, and recess periods.
       - **Target Audience:** Department, batch, semester, and cohort extraction.
       - **Required Action:** Active and passive modal directive extraction.
       - **Strict Rule:** NEVER invents or fabricates dates or details.
    4. **Rule-Based Priority Estimation Engine:**
       - **Urgent / High:** Imminent deadlines (< 48 hrs), punitive terms (`late fine`, `debarred`, `disciplinary action`), examination hall tickets, placement interview calls.
       - **Medium:** Scheduled course assignments, normal event registration deadlines.
       - **Low / Informational:** Campus holidays, general administrative advisories, informational circulars.
    5. **Verification & Anomaly Guardrails:**
       - **Missing Deadline Alert:** Triggered when action verbs are present but no calendar date is stipulated.
       - **Conflicting Dates Alert:** Triggered when late deadline precedes regular deadline or event date precedes registration.
       - **Ambiguity Detection:** Flagged when top 2 class probabilities have a small margin (< 0.15).
       - **Unsupported Content Guard:** Out-of-vocabulary or low confidence (< 0.22) automatically routed to `Unsupported / Low Confidence`.
    """)

    st.markdown("---")
    st.markdown("### 📋 Priority Estimation Decision Matrix")
    matrix_data = [
        {"Condition / Trigger": "Contains punitive penalty ('late fine', 'debarred', 'expulsion')", "Assigned Priority": "Urgent / High", "Example Impact": "Direct student financial or academic penalty"},
        {"Condition / Trigger": "Fee payment due date or arrears mentioned", "Assigned Priority": "Urgent / High", "Example Impact": "Financial compliance deadline"},
        {"Condition / Trigger": "Exam admit card / hall ticket collection or exam start", "Assigned Priority": "Urgent / High", "Example Impact": "Mandatory exam admission eligibility"},
        {"Condition / Trigger": "Placement drive interview shortlist or online test", "Assigned Priority": "Urgent / High", "Example Impact": "Time-sensitive career recruitment opportunity"},
        {"Condition / Trigger": "Assignment or project submission with stipulated date", "Assigned Priority": "Medium", "Example Impact": "Regular coursework milestone"},
        {"Condition / Trigger": "Event registration with scheduled deadline", "Assigned Priority": "Medium", "Example Impact": "Voluntary student activity compliance"},
        {"Condition / Trigger": "Holiday announcement or academic recess notice", "Assigned Priority": "Low / Informational", "Example Impact": "Passive notification of campus closure"},
        {"Condition / Trigger": "General campus discipline, ID card wear, cafeteria timing", "Assigned Priority": "Low / Informational", "Example Impact": "Campus operational awareness"},
    ]
    st.table(pd.DataFrame(matrix_data))
