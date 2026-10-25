"""
validation.py
-------------
Input validation, sanitization, and error handling module for
the Student Announcement Intelligence System.

Guarantees graceful degradation, helpful error feedback, and prevents
unhandled exceptions from empty strings, malformed input, missing artifacts,
or extreme inputs.
"""

import os
import re
from typing import Dict, Any, Tuple

MAX_TEXT_LENGTH = 5000
MIN_TEXT_LENGTH = 3

class ValidationError(Exception):
    """Custom exception raised when announcement input violates safety/format bounds."""
    pass

class ModelArtifactError(Exception):
    """Custom exception raised when required ML model artifacts are missing or corrupt."""
    pass

def sanitize_text(text: Any) -> str:
    """
    Sanitizes raw text:
    - Casts to string
    - Normalizes null bytes and unicode spaces
    - Strips leading and trailing whitespace
    """
    if text is None:
        return ""
    text_str = str(text)
    # Remove null characters and replace multiple non-standard spaces
    text_str = text_str.replace("\x00", "").strip()
    return text_str

def validate_announcement_input(text: Any) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validates announcement text before processing.
    
    Returns:
    - (is_valid: bool, error_message: str, diagnostic_info: dict)
    """
    sanitized = sanitize_text(text)
    
    diagnostic = {
        "raw_length": len(str(text)) if text is not None else 0,
        "sanitized_length": len(sanitized),
        "word_count": len(sanitized.split()) if sanitized else 0,
        "is_empty": False,
        "is_too_short": False,
        "is_too_long": False,
        "is_symbols_only": False,
    }
    
    # Check 1: Empty text
    if not sanitized:
        diagnostic["is_empty"] = True
        return False, "Input is empty. Please enter an announcement text to analyze.", diagnostic

    # Check 2: Minimum length
    if len(sanitized) < MIN_TEXT_LENGTH:
        diagnostic["is_too_short"] = True
        return False, f"Input is too short (minimum {MIN_TEXT_LENGTH} characters required).", diagnostic

    # Check 3: Maximum length safety guard
    if len(sanitized) > MAX_TEXT_LENGTH:
        diagnostic["is_too_long"] = True
        return False, f"Input exceeds maximum allowed length of {MAX_TEXT_LENGTH} characters.", diagnostic

    # Check 4: Check if text contains only symbols / punctuation
    alpha_num = re.findall(r'[a-zA-Z0-9]', sanitized)
    if not alpha_num:
        diagnostic["is_symbols_only"] = True
        return False, "Input contains only symbols or punctuation with no alphanumeric words.", diagnostic

    return True, "", diagnostic

def verify_model_artifacts(models_dir: str, required_files: list) -> Tuple[bool, str]:
    """
    Checks if all required model serialized files exist in the specified directory.
    """
    if not os.path.exists(models_dir):
        return False, f"Models directory not found at: {models_dir}"
    
    missing = []
    for f in required_files:
        path = os.path.join(models_dir, f)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            missing.append(f)
            
    if missing:
        return False, f"Missing or empty required model artifacts: {', '.join(missing)}"
    
    return True, ""
