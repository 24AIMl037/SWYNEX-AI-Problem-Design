@echo off
REM EduIntent AI - SWYNEX Task 1 & Task 2 Helper Script

if "%1"=="test" (
    echo [EduIntent AI] Running Automated Test Suite...
    python -m unittest tests/test_system.py
    exit /b %ERRORLEVEL%
)

if "%1"=="train" (
    echo [EduIntent AI] Retraining Machine Learning Models...
    python src/train.py
    exit /b %ERRORLEVEL%
)

if "%1"=="dataset" (
    echo [EduIntent AI] Regenerating Balanced Dataset...
    python create_dataset.py
    exit /b %ERRORLEVEL%
)

if "%1"=="pdf" (
    echo [EduIntent AI] Generating Executive PDF Report...
    python export_pdf_report.py
    exit /b %ERRORLEVEL%
)

echo [EduIntent AI] Starting Unified Streamlit Application (Task 1 & Task 2)...
streamlit run app.py
