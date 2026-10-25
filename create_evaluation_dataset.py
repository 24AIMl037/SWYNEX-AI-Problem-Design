"""
create_evaluation_dataset.py
----------------------------
Creates a comprehensive, manually annotated test suite for the Evaluation Lab.
Contains 30 diverse test cases across:
- Normal cases (each of the 7 categories)
- Ambiguous announcements (overlapping academic themes)
- Missing deadline cases (action requested without a date)
- Conflicting date cases (logically inconsistent dates)
- Out-of-domain / unrelated text (unsupported)
- Malformed inputs (empty, whitespace, symbols, extreme brevity)

Each test case contains:
- id: unique identifier
- test_type: 'normal', 'ambiguous', 'missing_deadline', 'conflicting_dates', 'out_of_domain', 'malformed'
- text: the announcement text
- expected_category: ground truth or 'Unsupported / Low Confidence'
- secondary_categories: possible valid secondary categories for ambiguous queries
- expected_deadlines: list of expected deadline dates/phrases
- expected_event_dates: list of expected event dates
- expected_audience: target audience or null
- expected_action: required action or null
- expected_priority: 'Urgent / High', 'Medium', 'Low / Informational'
- expected_flags: list of expected anomaly flags (e.g., 'missing_deadline', 'conflicting_dates', 'ambiguous', 'out_of_domain', 'malformed_input')
- description: human explanation of what this test case verifies
"""

import json
import os

TEST_CASES = [
    # -------------------------------------------------------------------------
    # 1. NORMAL CASES (7 samples - one per category)
    # -------------------------------------------------------------------------
    {
        "id": "NORM-01",
        "test_type": "normal",
        "text": "End Semester Theory Examinations for 5th and 7th semester B.Tech will commence from 15th December 2026. Hall tickets must be collected before 10th December 2026.",
        "expected_category": "Exams",
        "secondary_categories": [],
        "expected_deadlines": ["10th December 2026"],
        "expected_event_dates": ["15th December 2026"],
        "expected_audience": "5th and 7th semester B.Tech",
        "expected_action": "collected hall tickets",
        "expected_priority": "Urgent / High",
        "expected_flags": [],
        "description": "Standard exam notice with both commencement date and admit card collection deadline."
    },
    {
        "id": "NORM-02",
        "test_type": "normal",
        "text": "All 3rd year Computer Engineering students must submit Assignment 3 for Machine Learning on Google Classroom by 12th November 2026 before 11:59 PM.",
        "expected_category": "Assignments",
        "secondary_categories": [],
        "expected_deadlines": ["12th November 2026"],
        "expected_event_dates": [],
        "expected_audience": "3rd year Computer Engineering students",
        "expected_action": "submit Assignment 3 for Machine Learning",
        "expected_priority": "Medium",
        "expected_flags": [],
        "description": "Standard assignment submission notice with explicit target audience, LMS platform, and deadline."
    },
    {
        "id": "NORM-03",
        "test_type": "normal",
        "text": "Annual National Technical Symposium 'TECHNOVATION 2026' will be organized on 18th December 2026. All students must register before 5th December 2026 on the fest portal.",
        "expected_category": "Events",
        "secondary_categories": [],
        "expected_deadlines": ["5th December 2026"],
        "expected_event_dates": ["18th December 2026"],
        "expected_audience": "All students",
        "expected_action": "register on the fest portal",
        "expected_priority": "Medium",
        "expected_flags": [],
        "description": "Standard event notice with event date and participant registration deadline."
    },
    {
        "id": "NORM-04",
        "test_type": "normal",
        "text": "The college and administrative offices will remain closed from 11th November 2026 to 18th November 2026 on the auspicious occasion of Diwali festival. Classes resume on 19th November.",
        "expected_category": "Holidays",
        "secondary_categories": [],
        "expected_deadlines": [],
        "expected_event_dates": ["11th November 2026 to 18th November 2026"],
        "expected_audience": "college and administrative offices",
        "expected_action": None,
        "expected_priority": "Low / Informational",
        "expected_flags": [],
        "description": "Standard holiday circular specifying vacation period with no student action required."
    },
    {
        "id": "NORM-05",
        "test_type": "normal",
        "text": "Microsoft India campus placement drive for final year B.Tech CSE and IT students. Eligible candidates must upload their updated resume on the T&P portal before 15th November 2026.",
        "expected_category": "Placements",
        "secondary_categories": [],
        "expected_deadlines": ["15th November 2026"],
        "expected_event_dates": [],
        "expected_audience": "final year B.Tech CSE and IT students",
        "expected_action": "upload updated resume",
        "expected_priority": "Urgent / High",
        "expected_flags": [],
        "description": "Standard campus recruitment circular with resume submission deadline and eligibility criteria."
    },
    {
        "id": "NORM-06",
        "test_type": "normal",
        "text": "The last date for payment of Even Semester Tuition Fees is 30th November 2026. A late fine of Rs. 100 per day will be charged strictly after the due date.",
        "expected_category": "Fees",
        "secondary_categories": [],
        "expected_deadlines": ["30th November 2026"],
        "expected_event_dates": [],
        "expected_audience": None,
        "expected_action": "payment of Even Semester Tuition Fees",
        "expected_priority": "Urgent / High",
        "expected_flags": [],
        "description": "Standard fee notification with strict late fine penalty and deadline."
    },
    {
        "id": "NORM-07",
        "test_type": "normal",
        "text": "All students must wear their official college identity cards around their neck while inside campus premises. Security will strictly check ID cards at Gate 1.",
        "expected_category": "General",
        "secondary_categories": [],
        "expected_deadlines": [],
        "expected_event_dates": [],
        "expected_audience": "All students",
        "expected_action": "wear official college identity cards",
        "expected_priority": "Low / Informational",
        "expected_flags": [],
        "description": "Standard general administrative notice regarding campus discipline and ID cards."
    },

    # -------------------------------------------------------------------------
    # 2. AMBIGUOUS ANNOUNCEMENTS (5 samples)
    # -------------------------------------------------------------------------
    {
        "id": "AMBIG-01",
        "test_type": "ambiguous",
        "text": "Students with backlogs must pay the supplementary examination fee of Rs. 500 and submit the re-exam registration form before 18th November 2026.",
        "expected_category": "Exams",
        "secondary_categories": ["Fees"],
        "expected_deadlines": ["18th November 2026"],
        "expected_event_dates": [],
        "expected_audience": "Students with backlogs",
        "expected_action": "pay supplementary examination fee and submit form",
        "expected_priority": "Urgent / High",
        "expected_flags": ["ambiguous_category"],
        "description": "Ambiguous notice combining examination re-registration and fee payment keywords."
    },
    {
        "id": "AMBIG-02",
        "test_type": "ambiguous",
        "text": "The online placement screening examination for TCS recruitment will be conducted in the college examination hall on 22nd November 2026 from 10:00 AM.",
        "expected_category": "Placements",
        "secondary_categories": ["Exams"],
        "expected_deadlines": [],
        "expected_event_dates": ["22nd November 2026"],
        "expected_audience": None,
        "expected_action": "attend placement screening examination",
        "expected_priority": "Urgent / High",
        "expected_flags": ["ambiguous_category"],
        "description": "Cross-domain notice overlapping campus placement drive and examination hall test."
    },
    {
        "id": "AMBIG-03",
        "test_type": "ambiguous",
        "text": "Final year students presenting their Capstone Project Phase-1 must submit the project documentation file and clear all pending library dues before 28th November 2026.",
        "expected_category": "Assignments",
        "secondary_categories": ["Fees", "General"],
        "expected_deadlines": ["28th November 2026"],
        "expected_event_dates": [],
        "expected_audience": "Final year students",
        "expected_action": "submit project documentation and clear dues",
        "expected_priority": "Medium",
        "expected_flags": ["ambiguous_category"],
        "description": "Dual-intent notice mentioning capstone project submission alongside library dues clearance."
    },
    {
        "id": "AMBIG-04",
        "test_type": "ambiguous",
        "text": "Hackathon winners will receive direct summer internship placement interviews and certificates at the annual tech gala on 5th December 2026.",
        "expected_category": "Events",
        "secondary_categories": ["Placements"],
        "expected_deadlines": [],
        "expected_event_dates": ["5th December 2026"],
        "expected_audience": "Hackathon winners",
        "expected_action": None,
        "expected_priority": "Low / Informational",
        "expected_flags": ["ambiguous_category"],
        "description": "Notice overlapping tech hackathon event with placement internship rewards."
    },
    {
        "id": "AMBIG-05",
        "test_type": "ambiguous",
        "text": "Due to Diwali holiday week, the submission of Operating Systems Assignment 4 and lab records has been postponed to 23rd November 2026.",
        "expected_category": "Assignments",
        "secondary_categories": ["Holidays"],
        "expected_deadlines": ["23rd November 2026"],
        "expected_event_dates": [],
        "expected_audience": None,
        "expected_action": "submission of Operating Systems Assignment 4",
        "expected_priority": "Medium",
        "expected_flags": ["ambiguous_category"],
        "description": "Assignment deadline extension explicitly triggered by a festive holiday break."
    },

    # -------------------------------------------------------------------------
    # 3. MISSING DEADLINES (5 samples)
    # -------------------------------------------------------------------------
    {
        "id": "MISS-01",
        "test_type": "missing_deadline",
        "text": "All hostel students are urgently required to clear their outstanding mess fee arrears immediately. Failure to do so will result in room cancellation.",
        "expected_category": "Fees",
        "secondary_categories": [],
        "expected_deadlines": [],
        "expected_event_dates": [],
        "expected_audience": "hostel students",
        "expected_action": "clear outstanding mess fee arrears",
        "expected_priority": "Urgent / High",
        "expected_flags": ["missing_deadline"],
        "description": "Urgent demand for fee clearance with severe penalty threatened, but no deadline date specified."
    },
    {
        "id": "MISS-02",
        "test_type": "missing_deadline",
        "text": "All B.Tech students must submit their certified physics lab observation notebooks to the laboratory in-charge without delay.",
        "expected_category": "Assignments",
        "secondary_categories": [],
        "expected_deadlines": [],
        "expected_event_dates": [],
        "expected_audience": "All B.Tech students",
        "expected_action": "submit certified physics lab observation notebooks",
        "expected_priority": "Medium",
        "expected_flags": ["missing_deadline"],
        "description": "Action demanded (notebook submission) but without any specific due date or timestamp."
    },
    {
        "id": "MISS-03",
        "test_type": "missing_deadline",
        "text": "Collect your mid-term examination hall tickets from the departmental administrative counter during college working hours.",
        "expected_category": "Exams",
        "secondary_categories": [],
        "expected_deadlines": [],
        "expected_event_dates": [],
        "expected_audience": None,
        "expected_action": "collect mid-term examination hall tickets",
        "expected_priority": "Medium",
        "expected_flags": ["missing_deadline"],
        "description": "Exam hall ticket collection notice with action verb but missing collection deadline."
    },
    {
        "id": "MISS-04",
        "test_type": "missing_deadline",
        "text": "Eligible candidates shortlisted for the Amazon campus interview must upload their updated resume and GitHub profile URL on the training portal ASAP.",
        "expected_category": "Placements",
        "secondary_categories": [],
        "expected_deadlines": [],
        "expected_event_dates": [],
        "expected_audience": "Eligible candidates shortlisted for Amazon",
        "expected_action": "upload updated resume and GitHub profile URL",
        "expected_priority": "Urgent / High",
        "expected_flags": ["missing_deadline"],
        "description": "Placement action demanded 'ASAP' without a concrete calendar date or time limit."
    },
    {
        "id": "MISS-05",
        "test_type": "missing_deadline",
        "text": "Students who borrowed reference books from the department library are requested to return all books to the librarian.",
        "expected_category": "General",
        "secondary_categories": [],
        "expected_deadlines": [],
        "expected_event_dates": [],
        "expected_audience": "Students who borrowed reference books",
        "expected_action": "return all books to librarian",
        "expected_priority": "Low / Informational",
        "expected_flags": ["missing_deadline"],
        "description": "General book return request without any stipulated return deadline."
    },

    # -------------------------------------------------------------------------
    # 4. CONFLICTING DATES (4 samples)
    # -------------------------------------------------------------------------
    {
        "id": "CONF-01",
        "test_type": "conflicting_dates",
        "text": "The Annual Tech Hackathon 'CodeStorm' will be held on 10th November 2026. However, team registrations on the event portal will open on 20th November 2026.",
        "expected_category": "Events",
        "secondary_categories": [],
        "expected_deadlines": [],
        "expected_event_dates": ["10th November 2026", "20th November 2026"],
        "expected_audience": None,
        "expected_action": "team registrations",
        "expected_priority": "Medium",
        "expected_flags": ["conflicting_dates"],
        "description": "Temporal contradiction: Event date occurs before registration opening date."
    },
    {
        "id": "CONF-02",
        "test_type": "conflicting_dates",
        "text": "Assignment 2 submission deadline is 25th November 2026. Late submissions with 10% penalty will be accepted strictly until 15th November 2026.",
        "expected_category": "Assignments",
        "secondary_categories": [],
        "expected_deadlines": ["25th November 2026", "15th November 2026"],
        "expected_event_dates": [],
        "expected_audience": None,
        "expected_action": "Assignment 2 submission",
        "expected_priority": "Medium",
        "expected_flags": ["conflicting_dates"],
        "description": "Illogical deadline conflict: late submission cutoff is earlier than regular deadline."
    },
    {
        "id": "CONF-03",
        "test_type": "conflicting_dates",
        "text": "The college will observe a full public holiday for Diwali on 18th November 2026. All students must physically attend the Mid-Semester Examination in Block A on 18th November 2026.",
        "expected_category": "Exams",
        "secondary_categories": ["Holidays"],
        "expected_deadlines": [],
        "expected_event_dates": ["18th November 2026"],
        "expected_audience": "All students",
        "expected_action": "physically attend Mid-Semester Examination",
        "expected_priority": "Urgent / High",
        "expected_flags": ["conflicting_dates", "ambiguous_category"],
        "description": "Direct institutional conflict: Exam scheduled on a declared full campus holiday."
    },
    {
        "id": "CONF-04",
        "test_type": "conflicting_dates",
        "text": "Tuition fee payment last date without fine is 30th November 2026. Late fee charges begin from 20th November 2026 at Rs. 100 per day.",
        "expected_category": "Fees",
        "secondary_categories": [],
        "expected_deadlines": ["30th November 2026", "20th November 2026"],
        "expected_event_dates": [],
        "expected_audience": None,
        "expected_action": "Tuition fee payment",
        "expected_priority": "Urgent / High",
        "expected_flags": ["conflicting_dates"],
        "description": "Penalty start date (20th Nov) precedes the official regular due date (30th Nov)."
    },

    # -------------------------------------------------------------------------
    # 5. OUT-OF-DOMAIN / UNRELATED TEXT (4 samples)
    # -------------------------------------------------------------------------
    {
        "id": "OOD-01",
        "test_type": "out_of_domain",
        "text": "To bake a classic chocolate fudge cake, preheat the oven to 350 degrees Fahrenheit. Mix all-purpose flour, granulated sugar, dark cocoa powder, baking soda, and eggs.",
        "expected_category": "Unsupported / Low Confidence",
        "secondary_categories": [],
        "expected_deadlines": [],
        "expected_event_dates": [],
        "expected_audience": None,
        "expected_action": None,
        "expected_priority": "Low / Informational",
        "expected_flags": ["out_of_domain", "low_confidence"],
        "description": "Culinary recipe with zero relevance to college academic announcements."
    },
    {
        "id": "OOD-02",
        "test_type": "out_of_domain",
        "text": "The stock index dropped 350 points today as inflation worries weighed heavily on technology and banking equities across Asian markets.",
        "expected_category": "Unsupported / Low Confidence",
        "secondary_categories": [],
        "expected_deadlines": [],
        "expected_event_dates": [],
        "expected_audience": None,
        "expected_action": None,
        "expected_priority": "Low / Informational",
        "expected_flags": ["out_of_domain", "low_confidence"],
        "description": "Financial market update completely out of the academic announcements domain."
    },
    {
        "id": "OOD-03",
        "test_type": "out_of_domain",
        "text": "Hey bro what are you doing tonight? Let us go to the mall and watch the new Marvel movie then eat pepperoni pizza.",
        "expected_category": "Unsupported / Low Confidence",
        "secondary_categories": [],
        "expected_deadlines": [],
        "expected_event_dates": [],
        "expected_audience": None,
        "expected_action": None,
        "expected_priority": "Low / Informational",
        "expected_flags": ["out_of_domain", "low_confidence"],
        "description": "Casual personal messaging conversation containing no college administrative intent."
    },
    {
        "id": "OOD-04",
        "test_type": "out_of_domain",
        "text": "In yesterday's football match, the striker scored a hat-trick in the 85th minute, securing a 3-2 victory in the tournament final.",
        "expected_category": "Unsupported / Low Confidence",
        "secondary_categories": [],
        "expected_deadlines": [],
        "expected_event_dates": [],
        "expected_audience": None,
        "expected_action": None,
        "expected_priority": "Low / Informational",
        "expected_flags": ["out_of_domain", "low_confidence"],
        "description": "Sports match recap unrelated to campus notices."
    },

    # -------------------------------------------------------------------------
    # 6. MALFORMED & EDGE-CASE INPUTS (5 samples)
    # -------------------------------------------------------------------------
    {
        "id": "MALF-01",
        "test_type": "malformed",
        "text": "",
        "expected_category": "Unsupported / Low Confidence",
        "secondary_categories": [],
        "expected_deadlines": [],
        "expected_event_dates": [],
        "expected_audience": None,
        "expected_action": None,
        "expected_priority": "Low / Informational",
        "expected_flags": ["malformed_input", "empty_text"],
        "description": "Completely empty input string."
    },
    {
        "id": "MALF-02",
        "test_type": "malformed",
        "text": "    \t \n   ",
        "expected_category": "Unsupported / Low Confidence",
        "secondary_categories": [],
        "expected_deadlines": [],
        "expected_event_dates": [],
        "expected_audience": None,
        "expected_action": None,
        "expected_priority": "Low / Informational",
        "expected_flags": ["malformed_input", "whitespace_only"],
        "description": "Whitespace, tabs, and newline characters only."
    },
    {
        "id": "MALF-03",
        "test_type": "malformed",
        "text": "!@#$%^&*()_+{}|:\"<>?~`-=[]\\;',./",
        "expected_category": "Unsupported / Low Confidence",
        "secondary_categories": [],
        "expected_deadlines": [],
        "expected_event_dates": [],
        "expected_audience": None,
        "expected_action": None,
        "expected_priority": "Low / Informational",
        "expected_flags": ["malformed_input", "symbols_only"],
        "description": "Random special symbols and punctuation without any alphanumeric text."
    },
    {
        "id": "MALF-04",
        "test_type": "malformed",
        "text": "A",
        "expected_category": "Unsupported / Low Confidence",
        "secondary_categories": [],
        "expected_deadlines": [],
        "expected_event_dates": [],
        "expected_audience": None,
        "expected_action": None,
        "expected_priority": "Low / Informational",
        "expected_flags": ["malformed_input", "too_short"],
        "description": "Extremely short single-character string."
    },
    {
        "id": "MALF-05",
        "test_type": "malformed",
        "text": "urgent notice urgent notice urgent notice urgent notice urgent notice",
        "expected_category": "Unsupported / Low Confidence",
        "secondary_categories": ["General"],
        "expected_deadlines": [],
        "expected_event_dates": [],
        "expected_audience": None,
        "expected_action": None,
        "expected_priority": "Low / Informational",
        "expected_flags": ["low_confidence"],
        "description": "Repetitive keyword spamming devoid of informative academic context."
    }
]

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    json_path = os.path.join(dataset_dir, "evaluation_announcements.json")
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(TEST_CASES, f, indent=2)
        
    print(f"Created {len(TEST_CASES)} evaluation test cases.")
    print(f"Saved to: {json_path}")

if __name__ == "__main__":
    main()
