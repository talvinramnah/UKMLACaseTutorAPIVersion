import os
import random
from datetime import datetime, timedelta, timezone
from supabase import create_client

# --- CONFIGURATION ---
SUPABASE_URL = "https://wvfjgfqbfagxctyifhxu.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Ind2ZmpnZnFiZmFneGN0eWlmaHh1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDUwNzE2MzAsImV4cCI6MjA2MDY0NzYzMH0.TuPSRD2A-JfQ0cWDaqG28YIJO2ph1QY_1lRAzPHDvE8"
USER_ID = "66fe36bf-ea9e-43c9-9446-d97e0c83e4e0"  # Admin/test user
N_CASES = 15  # Number of mock cases to insert

# Example wards and conditions for variety
WARDS = [
    "Cardiology", "Respiratory", "Ent", "Paediatrics", "Ophthalmology"
]
CONDITIONS = [
    "Acute Myocardial Infarction", "Asthma", "Otitis Media", "Bronchiolitis", "Conjunctivitis"
]

# Example feedback
FEEDBACK_SUMMARIES = [
    "Good clinical reasoning, but missed a key investigation.",
    "Excellent management plan, but history could be more detailed.",
    "Solid case, but needs to clarify drug allergies.",
    "Great communication, but missed a red flag symptom."
]
POSITIVES = [
    ["Clear history taking", "Good differential diagnosis"],
    ["Identified key investigation", "Safe management plan"],
    ["Excellent communication", "Thorough examination"],
    ["Recognized urgent findings", "Escalated appropriately"]
]
IMPROVEMENTS = [
    ["Missed allergy check", "Could clarify drug doses"],
    ["History lacked detail", "Missed social context"],
    ["Did not ask about red flags", "Could improve structure"],
    ["Missed a key investigation", "Management plan incomplete"]
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