import os
import random
from datetime import datetime, timedelta, timezone
from supabase import create_client

# --- CONFIGURATION ---
SUPABASE_URL = "https://wvfjgfqbfagxctyifhxu.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Ind2ZmpnZnFiZmFneGN0eWlmaHh1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDUwNzE2MzAsImV4cCI6MjA2MDY0NzYzMH0.TuPSRD2A-JfQ0cWDaqG28YIJO2ph1QY_1lRAzPHDvE8"
USER_ID = "66fe36bf-ea9e-43c9-9446-d97e0c83e4e0"  # Admin/test user
N_CASES = 5  # Number of mock cases to insert

# Example wards and conditions for variety
WARDS = [
    "Cardiology", "Respiratory", "Ent", "Paediatrics", "Ophthalmology", "Infectious Diseases", "Dermatology"
]
CONDITIONS = [
    "Acute Myocardial Infarction", "Asthma", "Otitis Media", "Bronchiolitis", "Conjunctivitis", "Pneumonia", "Urinary Tract Infection", "Diabetes", "Hypertension", "Stroke", "Anxiety", "Depression"
]

# Example feedback
FEEDBACK_SUMMARIES = [
    "Good clinical reasoning demonstrated but missed some key investigations that would have helped confirm diagnosis.",
    "Excellent management plan and treatment choices, but patient history could be more detailed and structured.",
    "Solid case management overall, but needs to clarify drug allergies and medication history more thoroughly.",
    "Great communication skills and rapport with patient, but missed some important red flag symptoms to rule out.",
    "Appropriate initial assessment, but differential diagnosis could have been broader with more systematic approach.",
    "Strong diagnostic skills shown, however safety netting advice could have been more comprehensive.",
    "Good recognition of severity, but documentation of vital signs and clinical findings could be more detailed.",
    "Excellent prioritization of issues, though follow-up plan needs more specific timeframes and triggers."
]

POSITIVES = [
    ["Clear and systematic history taking approach", "Comprehensive differential diagnosis with good clinical reasoning"],
    ["Identified and ordered all key investigations promptly", "Safe and evidence-based management plan"],
    ["Excellent communication with clear explanation to patient", "Thorough physical examination with good technique"],
    ["Recognized urgent clinical findings quickly", "Appropriately escalated with clear SBAR handover"],
    ["Strong time management and prioritization", "Good documentation of clinical findings and plan"],
    ["Excellent patient-centered approach", "Appropriate use of clinical guidelines and protocols"],
    ["Clear safety-netting advice given", "Good recognition of clinical severity and risk"],
    ["Effective multidisciplinary team communication", "Thorough medication review and reconciliation"]
]

IMPROVEMENTS = [
    ["Did not complete allergy history check", "Could clarify specific drug doses and frequencies"],
    ["Patient history lacked important psychosocial context", "Missed key aspects of family history"],
    ["Did not systematically assess for red flag symptoms", "Clinical examination structure needs improvement"],
    ["Missed ordering some key baseline investigations", "Management plan missing specific review criteria"],
    ["Limited exploration of patient's ideas and concerns", "Documentation could be more detailed and structured"],
    ["Incomplete safety-netting advice given", "Follow-up plan needs clearer timeframes"],
    ["Did not fully assess treatment response", "Missing important aspects of systems review"],
    ["Communication with colleagues could be clearer", "Need to document clinical reasoning more explicitly"]
]
FOCUS_INSTRUCTIONS = [
    "For this case, focus ONLY on investigation. Do not ask or discuss management.",
    "For this case, focus ONLY on management. Do not ask or discuss investigation.",
    "No specific focus for this case."
]

# --- MAIN SCRIPT ---
def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Missing SUPABASE_URL or SUPABASE_KEY environment variables.")
        return
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    now = datetime.now(timezone.utc)
    for i in range(N_CASES):
        ward = random.choice(WARDS)
        condition = random.choice(CONDITIONS)
        feedback_summary = random.choice(FEEDBACK_SUMMARIES)
        feedback_positives = random.choice(POSITIVES)
        feedback_improvements = random.choice(IMPROVEMENTS)
        focus_instruction = random.choice(FOCUS_INSTRUCTIONS)
        result = random.choice([True, False])
        chat_transcript = [
            {"role": "user", "content": "Initial presentation."},
            {"role": "assistant", "content": "What is your next step?"},
            {"role": "user", "content": "Order ECG."},
            {"role": "assistant", "content": "Good, ECG shows ST elevation."}
        ]
        created_at = (now - timedelta(days=N_CASES - i)).isoformat()
        performance_data = {
            "user_id": USER_ID,
            "condition": condition,
            "result": result,
            "feedback_summary": feedback_summary,
            "feedback_positives": feedback_positives,
            "feedback_improvements": feedback_improvements,
            "chat_transcript": chat_transcript,
            "created_at": created_at,
            "ward": ward,
            "focus_instruction": focus_instruction
        }
        try:
            supabase.table("performance").insert(performance_data).execute()
            print(f"Inserted mock case {i+1}/{N_CASES}: {condition} ({ward})")
        except Exception as e:
            print(f"Failed to insert case {i+1}: {e}")

if __name__ == "__main__":
    main() 