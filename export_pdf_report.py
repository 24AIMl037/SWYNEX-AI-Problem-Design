import os
import json
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable

def generate_pdf():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(base_dir, "EduIntent_AI_Task1_Problem_Design.pdf")
    
    meta_path = os.path.join(base_dir, "models", "model_metadata.json")
    cm_img_path = os.path.join(base_dir, "reports", "confusion_matrix.png")
    mc_img_path = os.path.join(base_dir, "reports", "model_comparison.png")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1e3a8a'),
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#475569'),
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#0f766e'),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=6
    )

    gujarati_style = ParagraphStyle(
        'GujaratiBox',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor('#334155'),
        backColor=colors.HexColor('#f1f5f9'),
        borderColor=colors.HexColor('#cbd5e1'),
        borderWidth=1,
        borderPadding=6,
        spaceAfter=8
    )

    story = []

    # Title & Header
    story.append(Paragraph("Task 1 – AI Problem Design Document", title_style))
    story.append(Paragraph("Project: EduIntent AI – Intelligent Student Query Classification System", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1e3a8a'), spaceAfter=10))

    # 1. Executive Summary Table
    summary_data = [
        [Paragraph("<b>Parameter</b>", body_style), Paragraph("<b>Specification / Result</b>", body_style)],
        [Paragraph("<b>Project Name</b>", body_style), Paragraph("EduIntent AI – Student Query Intent Classifier", body_style)],
        [Paragraph("<b>AI Task</b>", body_style), Paragraph("Multi-Class Text Intent Classification", body_style)],
        [Paragraph("<b>Unique Features</b>", body_style), Paragraph("Explainable AI (XAI), Ambiguity Detection, Low Confidence Guardrails, Code-Mixed Language Support", body_style)],
        [Paragraph("<b>Academic Categories</b>", body_style), Paragraph("9 Core Computer Science & Administrative Subjects", body_style)],
        [Paragraph("<b>Dataset Size</b>", body_style), Paragraph("450 Hand-curated Questions (50 per category)", body_style)],
        [Paragraph("<b>Best Algorithm</b>", body_style), Paragraph("Calibrated Linear SVM (TF-IDF Unigrams + Bigrams)", body_style)],
        [Paragraph("<b>Test Accuracy</b>", body_style), Paragraph("<b>87.78%</b> (Target: ≥ 85%)", body_style)],
        [Paragraph("<b>Macro F1-Score</b>", body_style), Paragraph("<b>0.8746</b> (Target: ≥ 0.80)", body_style)],
        [Paragraph("<b>5-Fold CV Accuracy</b>", body_style), Paragraph("<b>89.72%</b> (± 0.02)", body_style)]
    ]

    t_summary = Table(summary_data, colWidths=[140, 390])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#e2e8f0')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 8))

    # 2. Problem Statement
    story.append(Paragraph("1. Problem Statement & Unique System Design", h1_style))
    p1_eng = """
    Students frequently ask academic queries across subjects (Python, ML, Deep Learning, DBMS, OS, Networks, Flutter, DAA, General Academic Queries). 
    Manually categorizing queries is inefficient. EduIntent AI automatically categorizes queries, displays confidence scores, extracts key contributing terms (XAI), detects ambiguous cross-domain queries, and alerts users on low-confidence inputs.
    """
    story.append(Paragraph(p1_eng, body_style))

    p1_guj = """
    <b>Gujarati Context:</b><br/>
    Students દરરોજ academic queries પૂછે છે. EduIntent AI model query સમજીને automatically યોગ્ય category, confidence score, અને key terms આપશે.
    """
    story.append(Paragraph(p1_guj, gujarati_style))

    # 3. Model Evaluation & Benchmarks Table
    story.append(Paragraph("2. Benchmark Comparison Across Algorithms", h1_style))
    bench_data = [
        [Paragraph("<b>Model Algorithm</b>", body_style), Paragraph("<b>Test Accuracy</b>", body_style), Paragraph("<b>Macro F1</b>", body_style), Paragraph("<b>5-Fold CV</b>", body_style)]
    ]

    for m_name, metrics in meta["results_summary"].items():
        bench_data.append([
            Paragraph(f"<b>{m_name}</b>", body_style),
            Paragraph(f"{metrics['Accuracy']*100:.2f}%", body_style),
            Paragraph(f"{metrics['F1-Score (Macro)']:.4f}", body_style),
            Paragraph(f"{metrics['CV_Mean_Accuracy']*100:.2f}%", body_style)
        ])

    t_bench = Table(bench_data, colWidths=[180, 110, 110, 130])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#94a3b8')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_bench)
    story.append(Spacer(1, 8))

    # 4. Error Analysis Table
    story.append(Paragraph("3. Qualitative Error Analysis Table", h1_style))
    err_data = [
        [Paragraph("<b>Question</b>", body_style), Paragraph("<b>Expected</b>", body_style), Paragraph("<b>Predicted</b>", body_style), Paragraph("<b>Reason / Analysis</b>", body_style)],
        [Paragraph("<i>'Neural network optimization loss function'</i>", body_style), Paragraph("Deep Learning", body_style), Paragraph("Machine Learning", body_style), Paragraph("Shared ML/DL vocabulary (loss functions, optimization)", body_style)],
        [Paragraph("<i>'B+ Tree structure in file indexing'</i>", body_style), Paragraph("DBMS", body_style), Paragraph("DAA / Algorithms", body_style), Paragraph("Shared data structures & indexing terminology", body_style)],
        [Paragraph("<i>'How to plot correlation matrix in Pandas?'</i>", body_style), Paragraph("Python/Data Sci", body_style), Paragraph("Python/Data Sci", body_style), Paragraph("✅ Correct (High precision keyword match)", body_style)]
    ]
    t_err = Table(err_data, colWidths=[160, 95, 95, 180])
    t_err.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f766e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_err)
    story.append(Spacer(1, 8))

    # Add images if available
    if os.path.exists(mc_img_path):
        story.append(Paragraph("<b>Figure 1: Benchmark Comparison Chart</b>", body_style))
        story.append(Image(mc_img_path, width=480, height=220))
        story.append(Spacer(1, 6))

    if os.path.exists(cm_img_path):
        story.append(Paragraph("<b>Figure 2: Confusion Matrix (Linear SVM)</b>", body_style))
        story.append(Image(cm_img_path, width=420, height=320))
        story.append(Spacer(1, 6))

    doc.build(story)
    print(f"PDF Report successfully generated at '{pdf_path}'.")

if __name__ == "__main__":
    generate_pdf()
