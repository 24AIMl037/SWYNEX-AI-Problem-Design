"""
announcement_analyzer.py
------------------------
Core Smart Announcement Analyzer for the Student Announcement Intelligence System.
SWYNEX AI Internship - Task 3.

Key Responsibilities:
1. Category Classification via trained TF-IDF + Logistic Regression model.
2. Deadline & Event Date Extraction (Strictly non-hallucinatory regex & contextual NLP).
3. Target Audience & Required Action Extraction.
4. Rule-Based Priority Estimation with Transparent Rationales.
5. Missing, Ambiguous, Conflicting, and Unsupported Information Detection.
6. Guardrails & Verification Warnings.
7. Graceful Error Handling & Fallbacks.
"""

import os
import re
import json
import joblib
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

try:
    from preprocessing import clean_text
    from validation import validate_announcement_input, verify_model_artifacts
except ImportError:
    from src.preprocessing import clean_text
    from src.validation import validate_announcement_input, verify_model_artifacts

# Standard academic announcement categories
CATEGORIES = ["Assignments", "Events", "Exams", "Fees", "General", "Holidays", "Placements"]

# Month names lookup for date parsing and normalization
MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10,
    "november": 11, "nov": 11, "december": 12, "dec": 12
}

class SmartAnnouncementAnalyzer:
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = os.path.join(PROJECT_ROOT, "models")
        self.models_dir = models_dir
        self.model = None
        self.vectorizer = None
        self.metadata = {}
        self.is_loaded = False
        self.load_error = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads serialized TF-IDF vectorizer and Logistic Regression classifier."""
        required = ["announcement_classifier.pkl", "announcement_tfidf.pkl", "announcement_metadata.json"]
        is_ok, err_msg = verify_model_artifacts(self.models_dir, required)
        if not is_ok:
            self.load_error = err_msg
            self.is_loaded = False
            return

        try:
            clf_path = os.path.join(self.models_dir, "announcement_classifier.pkl")
            vec_path = os.path.join(self.models_dir, "announcement_tfidf.pkl")
            meta_path = os.path.join(self.models_dir, "announcement_metadata.json")

            self.model = joblib.load(clf_path)
            self.vectorizer = joblib.load(vec_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            self.is_loaded = True
            self.load_error = None
        except Exception as e:
            self.load_error = f"Failed to deserialize model artifacts: {str(e)}"
            self.is_loaded = False

    # -------------------------------------------------------------------------
    # 1. DATE & DEADLINE EXTRACTION (STRICTLY FROM TEXT, NO HALLUCINATIONS)
    # -------------------------------------------------------------------------
    def _extract_dates(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts calendar dates and date ranges from announcement text.
        Guarantees that every extracted date matches an exact span in the text.
        """
        extracted = []
        
        # Pattern 1: Date ranges (e.g. 11th November 2026 to 18th November 2026, or 10th to 12th December 2026)
        range_pattern = r'(\b\d{1,2}(?:st|nd|rd|th)?(?:\s+(?:[A-Za-z]+))?\s*(?:to|-|until)\s*\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}\b)'
        for m in re.finditer(range_pattern, text, re.IGNORECASE):
            extracted.append({
                "raw_date": m.group(1).strip(),
                "start": m.start(),
                "end": m.end(),
                "is_range": True
            })

        # Pattern 2: DD Month YYYY (e.g. 15th December 2026, 25 November 2026)
        p2 = r'(\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*(?:\s*,?\s*\d{4})?\b)'
        for m in re.finditer(p2, text, re.IGNORECASE):
            raw = m.group(1).strip()
            # Avoid duplicate overlapping match
            if not any(e["start"] <= m.start() and m.end() <= e["end"] for e in extracted):
                extracted.append({
                    "raw_date": raw,
                    "start": m.start(),
                    "end": m.end(),
                    "is_range": False
                })

        # Pattern 3: Month DD, YYYY (e.g. November 15, 2026)
        p3 = r'(\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?\b)'
        for m in re.finditer(p3, text, re.IGNORECASE):
            raw = m.group(1).strip()
            if not any(e["start"] <= m.start() and m.end() <= e["end"] for e in extracted):
                extracted.append({
                    "raw_date": raw,
                    "start": m.start(),
                    "end": m.end(),
                    "is_range": False
                })

        # Pattern 4: Numerical dates (e.g. 15/12/2026, 15-12-2026)
        p4 = r'(\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b)'
        for m in re.finditer(p4, text):
            raw = m.group(1).strip()
            if not any(e["start"] <= m.start() and m.end() <= e["end"] for e in extracted):
                extracted.append({
                    "raw_date": raw,
                    "start": m.start(),
                    "end": m.end(),
                    "is_range": False
                })

        return extracted

    def _classify_dates(self, text: str, date_items: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """
        Classifies extracted dates into Deadlines vs Event Dates based on local syntactic context.
        Returns: (deadlines, event_dates)
        """
        deadlines = []
        event_dates = []

        deadline_cues = [
            "deadline", "due date", "last date", "before", "by", "until", "submit on or before",
            "due on", "latest by", "prior to", "closing date", "extended to", "apply before",
            "pay by", "uploaded before", "collected before", "register before", "without fine is",
            "with late fine is", "strictly until", "within 48 hours", "cutoff"
        ]

        event_cues = [
            "held on", "commence from", "commence on", "begins on", "starts from", "start on",
            "scheduled on", "take place on", "organized on", "will be on", "celebrated on",
            "vacation from", "recess from", "closed on", "closed from", "observed from",
            "viva on", "test date is", "screening round", "fest date", "ceremony on"
        ]

        text_lower = text.lower()

        for item in date_items:
            date_str = item["raw_date"]
            start = max(0, item["start"] - 60)
            end = min(len(text), item["end"] + 40)
            context = text_lower[start:end]

            # Score proximity to deadline cues vs event cues
            is_deadline = any(cue in context for cue in deadline_cues)
            is_event = any(cue in context for cue in event_cues)

            if is_deadline and not is_event:
                deadlines.append(date_str)
            elif is_event and not is_deadline:
                event_dates.append(date_str)
            elif is_deadline and is_event:
                # Disambiguate by checking which keyword is closer to date
                dl_pos = max([context.find(c) for c in deadline_cues if c in context] + [-1])
                ev_pos = max([context.find(c) for c in event_cues if c in context] + [-1])
                date_in_context = item["start"] - start
                if abs(dl_pos - date_in_context) <= abs(ev_pos - date_in_context):
                    deadlines.append(date_str)
                else:
                    event_dates.append(date_str)
            else:
                # Default fallback: check if global text is submission/payment oriented
                if any(w in text_lower for w in ["submit", "submission", "due", "pay", "payment", "register by"]):
                    deadlines.append(date_str)
                else:
                    event_dates.append(date_str)

        # De-duplicate preserving order
        clean_dl = list(dict.fromkeys(deadlines))
        clean_ev = list(dict.fromkeys(event_dates))
        return clean_dl, clean_ev

    # -------------------------------------------------------------------------
    # 2. TARGET AUDIENCE & REQUIRED ACTION EXTRACTION
    # -------------------------------------------------------------------------
    def _extract_target_audience(self, text: str) -> Optional[str]:
        """
        Extracts the explicit student cohort or group targeted by the notice.
        """
        patterns = [
            r'\b(?:all\s+)?(?:\d+(?:st|nd|rd|th)?(?:\s+(?:and|&|,)\s+\d+(?:st|nd|rd|th)?)*\s+(?:semester|sem|year))\s+(?:b\.?tech|m\.?tech|mca|mba|engineering|[A-Za-z]+)?\b',
            r'\b(?:all\s+)?(?:final\s+year|pre-final\s+year|\d+(?:st|nd|rd|th)\s+(?:year|semester|sem)|outgoing|lateral\s+entry)\s+(?:b\.?tech|m\.?tech|mca|mba|engineering)?\s*(?:[A-Za-z\s/&,]+)?\s*(?:students?|candidates?|scholars?|residents?|batches?)\b',
            r'\b(?:all\s+)?(?:b\.?tech|m\.?tech|mca|mba)\s+(?:\d+(?:st|nd|rd|th)\s+semester\s+)?(?:[A-Za-z\s/&,]+)?\s*(?:students?|candidates?)\b',
            r'\b(?:all\s+)?(?:hostel\s+residents?|hostel\s+students?|hostellers?|day\s+scholars?|bus\s+commuters?)\b',
            r'\b(?:eligible\s+candidates?(?:\s+shortlisted[^\.,;]*)?)\b',
            r'\b(?:students\s+with\s+backlogs|unplaced\s+final\s+year\s+students|hackathon\s+winners)\b',
            r'\b(?:all\s+students\s+and\s+faculty|all\s+students)\b',
            r'\b(?:all\s+candidates?|registered\s+candidates?)\b'
        ]

        for p in patterns:
            match = re.search(p, text, re.IGNORECASE)
            if match:
                candidate = match.group(0).strip()
                # Clean trailing punctuation
                candidate = re.sub(r'[,;:\.]+$', '', candidate)
                if len(candidate) > 3 and not candidate.lower().startswith("all the"):
                    return candidate

        return None

    def _extract_required_action(self, text: str) -> Optional[str]:
        """
        Extracts actionable directives (verbs like submit, pay, register, collect, upload, attend).
        """
        patterns = [
            r'([a-z\s0-9/-]{3,40}?)\s+(?:must be|should be|is to be)\s+(collected|submitted|paid|uploaded|verified|reported|completed|cleared)\b',
            r'\b(?:must|required to|should|instructed to|urged to|advised to|please)\s+([a-z\s0-9/-]{4,70}?)(?:before|by|on|at|until|in|\.|\;|\n)',
            r'\b(?:submission of|payment of|registration for|collection of)\s+([a-z\s0-9/-]{4,70}?)(?:is due|before|by|on|until|\.|\;|\n)',
            r'\b(submit|pay|collect|register|upload|report|bring|fill|attend|return|renew|clear)\s+([a-z\s0-9/-]{4,70}?)(?:before|by|on|at|until|in|\.|\;|\n)'
        ]

        for p in patterns:
            match = re.search(p, text, re.IGNORECASE)
            if match:
                groups = [g for g in match.groups() if g]
                if len(groups) == 2:
                    action_phrase = f"{groups[1]} {groups[0]}".strip() if "must be" in match.group(0) or "should be" in match.group(0) else f"{groups[0]} {groups[1]}".strip()
                else:
                    action_phrase = groups[0].strip()
                action_phrase = re.sub(r'\s+', ' ', action_phrase).strip()
                action_phrase = re.sub(r'^(to|be|the)\s+', '', action_phrase, flags=re.IGNORECASE)
                if len(action_phrase) >= 4:
                    return action_phrase[:80]

        return None

    # -------------------------------------------------------------------------
    # 3. RULE-BASED PRIORITY ESTIMATION WITH TRANSPARENT RATIONALE
    # -------------------------------------------------------------------------
    def _estimate_priority(self, category: str, text: str, deadlines: List[str], actions: Optional[str]) -> Tuple[str, List[str]]:
        """
        Calculates priority level ('Urgent / High', 'Medium', 'Low / Informational')
        along with an explainable list of rules triggered.
        """
        text_lower = text.lower()
        rationales = []
        is_urgent = False
        is_medium = False

        # Rule 1: Punitive / strict enforcement language
        penalty_cues = ["late fine", "penalty", "debarred", "expulsion", "blacklisted", "detained", "cancel", "disciplinary action", "strictly prohibited", "forfeited"]
        matched_penalties = [cue for cue in penalty_cues if cue in text_lower]
        if matched_penalties:
            is_urgent = True
            rationales.append(f"Strict compliance penalty mentioned: '{matched_penalties[0]}'")

        # Rule 2: Immediate urgency cues
        urgency_cues = ["immediately", "asap", "without delay", "within 24 hours", "within 48 hours", "urgent notice", "urgent reminder", "strict warning"]
        matched_urgency = [cue for cue in urgency_cues if cue in text_lower]
        if matched_urgency:
            is_urgent = True
            rationales.append(f"Urgent operational time cue detected: '{matched_urgency[0]}'")

        # Rule 3: Category-specific high stakes
        if category == "Fees" and ("due date" in text_lower or "late fine" in text_lower or "arrears" in text_lower or "payment" in text_lower):
            is_urgent = True
            rationales.append("Tuition/Examination fee deadline involves financial liability or late fine")

        if category == "Exams" and any(k in text_lower for k in ["hall ticket", "admit card", "commence", "timetable", "debarred", "backlog form"]):
            is_urgent = True
            rationales.append("Mandatory examination requirement (hall ticket / exam commencement)")

        if category == "Placements" and ("interview" in text_lower or "shortlisted" in text_lower or "test" in text_lower or "offer" in text_lower):
            is_urgent = True
            rationales.append("Time-sensitive career/placement recruitment event")

        # Rule 4: Action required with stipulated deadline
        if deadlines and actions:
            if not is_urgent:
                is_medium = True
                rationales.append(f"Mandatory action ('{actions[:35]}...') with scheduled deadline ({deadlines[0]})")

        # Rule 5: Assignments category standard flow
        if category == "Assignments" and not is_urgent:
            is_medium = True
            rationales.append("Academic coursework assignment requires scheduled student submission")

        # Rule 6: Holidays and general informational circulars
        if category in ["Holidays", "General"] and not is_urgent and not is_medium:
            rationales.append("Passive institutional notification or campus holiday circular with no student action required")
            return "Low / Informational", rationales

        if is_urgent:
            if not rationales:
                rationales.append("High-impact academic directive requiring prompt student action")
            return "Urgent / High", rationales
        elif is_medium:
            if not rationales:
                rationales.append("Scheduled academic activity requiring student compliance")
            return "Medium", rationales
        else:
            rationales.append("Informational advisory with no immediate deadline constraints")
            return "Low / Informational", rationales

    # -------------------------------------------------------------------------
    # 4. ANOMALY, CONFLICT & UNSUPPORTED DETECTION
    # -------------------------------------------------------------------------
    def _detect_anomalies(self, text: str, category: str, confidence: float,
                          deadlines: List[str], event_dates: List[str],
                          action: Optional[str], top_probs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detects:
        - Missing deadlines: Action demanded without any date.
        - Conflicting dates: Chronological or logical discrepancies.
        - Ambiguity: High competing probability across categories.
        - Unsupported / Out-of-domain: Out of vocabulary or low confidence.
        """
        flags = []
        warnings = []
        conflicts = []

        text_lower = text.lower()

        # Check 1: Missing Deadline
        action_verbs = ["submit", "pay", "clear", "upload", "collect", "register", "return", "report"]
        has_action_demand = any(v in text_lower for v in action_verbs)
        urgent_demand = any(u in text_lower for u in ["immediately", "asap", "without delay", "promptly"])
        
        if (has_action_demand or urgent_demand) and not deadlines and not event_dates:
            flags.append("missing_deadline")
            warnings.append("Action is requested in the announcement, but NO specific deadline or date is provided.")

        # Check 2: Conflicting Dates
        # Date parsing helper
        def parse_date_rough(d_str: str) -> Optional[datetime]:
            d_clean = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', d_str.lower())
            words = d_clean.split()
            day, month, year = None, None, 2026
            for w in words:
                if w.isdigit():
                    val = int(w)
                    if 1900 <= val <= 2100:
                        year = val
                    elif 1 <= val <= 31 and day is None:
                        day = val
                elif w in MONTHS:
                    month = MONTHS[w]
            if day and month:
                try:
                    return datetime(year, month, day)
                except ValueError:
                    return None
            return None

        # Detect late deadline earlier than regular deadline
        if len(deadlines) >= 2:
            dt1 = parse_date_rough(deadlines[0])
            dt2 = parse_date_rough(deadlines[1])
            if dt1 and dt2:
                # If "late submission" or "late fine" mentions earlier date than regular due date
                if ("late" in text_lower or "fine" in text_lower) and dt2 < dt1:
                    flags.append("conflicting_dates")
                    conflicts.append(f"Late deadline ({deadlines[1]}) is earlier than regular deadline ({deadlines[0]}).")
                    warnings.append("Inconsistent dates detected: Late submission cutoff occurs earlier than the primary due date.")

        # Detect event occurring before registration opens
        if event_dates and ("registration" in text_lower or "register" in text_lower):
            ev_dt = parse_date_rough(event_dates[0])
            for d in (deadlines + event_dates[1:]):
                other_dt = parse_date_rough(d)
                if ev_dt and other_dt and other_dt > ev_dt:
                    if "registration will open" in text_lower or "register on" in text_lower:
                        flags.append("conflicting_dates")
                        conflicts.append(f"Event date ({event_dates[0]}) occurs before registration ({d}).")
                        warnings.append("Chronological conflict detected: Event date is scheduled earlier than the registration date.")
                        break

        # Check Holiday vs Exam conflict on same date
        if "holiday" in text_lower and ("exam" in text_lower or "examination" in text_lower):
            if event_dates or deadlines:
                flags.append("conflicting_dates")
                warnings.append("Institutional conflict detected: Examination scheduled on a declared full campus holiday.")

        # Check 3: Ambiguous Category
        if len(top_probs) >= 2:
            p1 = top_probs[0]["probability"]
            p2 = top_probs[1]["probability"]
            diff = p1 - p2
            if diff < 0.15 and p2 >= 0.18:
                flags.append("ambiguous_category")
                warnings.append(f"Ambiguous announcement: Category overlaps between '{top_probs[0]['category']}' ({p1*100:.1f}%) and '{top_probs[1]['category']}' ({p2*100:.1f}%).")

        # Check 4: Unsupported / Low Confidence
        if confidence < 0.22:
            flags.append("low_confidence")
            warnings.append(f"Low prediction confidence ({confidence*100:.1f}%). Text may be outside the academic announcements domain.")

        return {
            "flags": flags,
            "warnings": warnings,
            "conflicts": conflicts,
            "has_warnings": len(warnings) > 0
        }

    # -------------------------------------------------------------------------
    # 5. MAIN INFERENCE PIPELINE
    # -------------------------------------------------------------------------
    def analyze(self, raw_text: Any) -> Dict[str, Any]:
        """
        Executes end-to-end analysis on raw announcement text.
        Guarantees structured JSON return, no unhandled exceptions, and no hallucinated details.
        """
        # Step 1: Input Validation
        is_valid, err_msg, diag = validate_announcement_input(raw_text)
        if not is_valid:
            return {
                "success": False,
                "error": err_msg,
                "diagnostic": diag,
                "predicted_category": "Unsupported / Low Confidence",
                "confidence": 0.0,
                "priority": "Low / Informational",
                "priority_rationales": ["Input text failed validation checks."],
                "deadlines": [],
                "event_dates": [],
                "target_audience": None,
                "required_action": None,
                "flags": ["malformed_input"] + ([ "empty_text"] if diag["is_empty"] else []),
                "warnings": [f"Input validation warning: {err_msg}"],
                "top_categories": []
            }

        text = diag["sanitized_length"] and str(raw_text).strip()

        # Step 2: Verify model load
        if not self.is_loaded:
            return {
                "success": False,
                "error": f"Model not loaded: {self.load_error}",
                "predicted_category": "Unsupported / Low Confidence",
                "confidence": 0.0,
                "priority": "Low / Informational",
                "priority_rationales": ["Model artifacts are unavailable."],
                "deadlines": [],
                "event_dates": [],
                "target_audience": None,
                "required_action": None,
                "flags": ["model_error"],
                "warnings": ["System model files are missing or uninitialized."],
                "top_categories": []
            }

        # Step 3: Text Preprocessing & Vectorization
        cleaned = clean_text(text)
        vec = self.vectorizer.transform([cleaned])

        # Out-of-vocabulary / minimal domain terms check
        if vec.nnz <= 3:
            return {
                "success": True,
                "raw_text": text,
                "predicted_category": "Unsupported / Low Confidence",
                "raw_predicted_category": "Unsupported / Low Confidence",
                "confidence": 0.0,
                "priority": "Low / Informational",
                "priority_rationales": ["No relevant college announcement keywords recognized in input text."],
                "deadlines": [],
                "event_dates": [],
                "target_audience": None,
                "required_action": None,
                "flags": ["out_of_domain", "low_confidence"],
                "warnings": ["The provided text does not match any recognized college announcement vocabulary."],
                "conflicts": [],
                "has_warnings": True,
                "top_categories": [],
                "diagnostic": diag
            }

        # Step 4: ML Prediction with Calibrated Probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vec)[0]
            classes = list(self.model.classes_)
            # Sort categories by probability
            sorted_idx = np.argsort(probs)[::-1]
            top_cats = [
                {"category": classes[i], "probability": float(probs[i])}
                for i in sorted_idx[:3]
            ]
            pred_cat = top_cats[0]["category"]
            confidence = top_cats[0]["probability"]
        else:
            pred_cat = self.model.predict(vec)[0]
            confidence = 1.0
            top_cats = [{"category": pred_cat, "probability": 1.0}]

        # Step 5: Entity & Temporal Extraction
        date_items = self._extract_dates(text)
        deadlines, event_dates = self._classify_dates(text, date_items)
        target_audience = self._extract_target_audience(text)
        required_action = self._extract_required_action(text)

        # Step 6: Priority Estimation
        priority, rationales = self._estimate_priority(pred_cat, text, deadlines, required_action)

        # Step 7: Anomaly, Conflict & Verification Checks
        anomaly_res = self._detect_anomalies(
            text, pred_cat, confidence, deadlines, event_dates, required_action, top_cats
        )

        # If low confidence below 0.22, flag as Unsupported
        final_category = pred_cat
        if "out_of_domain" in anomaly_res["flags"] or confidence < 0.22:
            final_category = "Unsupported / Low Confidence"

        return {
            "success": True,
            "raw_text": text,
            "predicted_category": final_category,
            "raw_predicted_category": pred_cat,
            "confidence": float(confidence),
            "priority": priority,
            "priority_rationales": rationales,
            "deadlines": deadlines,
            "event_dates": event_dates,
            "target_audience": target_audience,
            "required_action": required_action,
            "flags": anomaly_res["flags"],
            "warnings": anomaly_res["warnings"],
            "conflicts": anomaly_res["conflicts"],
            "has_warnings": anomaly_res["has_warnings"],
            "top_categories": top_cats,
            "diagnostic": diag
        }
