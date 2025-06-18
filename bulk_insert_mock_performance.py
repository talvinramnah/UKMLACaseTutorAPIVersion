import os
import random
import uuid
import time
from datetime import datetime, timedelta, timezone
from supabase import create_client

# --- CONFIGURATION ---
SUPABASE_URL = "https://wvfjgfqbfagxctyifhxu.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Ind2ZmpnZnFiZmFneGN0eWlmaHh1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDUwNzE2MzAsImV4cCI6MjA2MDY0NzYzMH0.TuPSRD2A-JfQ0cWDaqG28YIJO2ph1QY_1lRAzPHDvE8"
N_USERS_PER_SCHOOL = 12  # Ensure >=10 for leaderboard
N_SCHOOLS = 3
N_CASES_PER_USER = 8

SCHOOLS = [
    "University of Bristol Medical School",
    "University of Oxford Medical School",
    "University of Cambridge Medical School"
]
YEAR_GROUPS = ["1st year", "2nd year", "3rd year", "4th year", "5th year"]

CONDITIONS = [
    "Acute Myocardial Infarction", "Asthma", "Otitis Media", "Bronchiolitis", "Conjunctivitis", "Pneumonia", "Urinary Tract Infection", "Diabetes", "Hypertension", "Stroke", "Anxiety", "Depression"
]
WARDS = [
    "Cardiology", "Respiratory", "Ent", "Paediatrics", "Ophthalmology", "Infectious Diseases", "Dermatology"
]
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
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    now = datetime.now(timezone.utc)
    user_ids = []
    for school_idx, school in enumerate(SCHOOLS[:N_SCHOOLS]):
        for i in range(N_USERS_PER_SCHOOL):
            anon_username = f"medmock{school_idx}{i}"
            email = f"{anon_username}@mock.com"
            year_group = random.choice(YEAR_GROUPS)
            # Try to sign up user (idempotent)
            try:
                result = supabase.auth.sign_up({
                    "email": email,
                    "password": "TestPassword123!"
                })
                user_id = result.user.id
                print(f"Signed up user: {anon_username} ({email})")
                time.sleep(0.2)  # Avoid rate limits
            except Exception as e:
                # If user already exists, fetch user_id
                print(f"User {anon_username} already exists or signup failed: {e}")
                # Try to get user_id from user_metadata
                exists = supabase.table("user_metadata").select("user_id").eq("anon_username", anon_username).execute()
                if exists.data:
                    user_id = exists.data[0]["user_id"]
                else:
                    print(f"Could not get user_id for {anon_username}, skipping.")
                    continue
            # Insert user_metadata (idempotent)
            exists = supabase.table("user_metadata").select("user_id").eq("anon_username", anon_username).execute()
            if exists.data:
                print(f"User_metadata for {anon_username} already exists, skipping user_metadata insert.")
            else:
                supabase.table("user_metadata").insert({
                    "user_id": user_id,
                    "anon_username": anon_username,
                    "med_school": school,
                    "year_group": year_group,
                    "name": f"Mock Name {anon_username}"
                }).execute()
                print(f"Inserted user_metadata: {anon_username} ({school}, {year_group})")
            user_ids.append((user_id, anon_username, school))
            # Insert performance cases
            for j in range(N_CASES_PER_USER):
                ward = random.choice(WARDS)
                condition = random.choice(CONDITIONS)
                feedback_summary = random.choice(FEEDBACK_SUMMARIES)
                feedback_positives = random.choice(POSITIVES)
                feedback_improvements = random.choice(IMPROVEMENTS)
                focus_instruction = random.choice(FOCUS_INSTRUCTIONS)
                result_case = random.choice([True, False])
                chat_transcript = [
                    {"role": "user", "content": "Initial presentation."},
                    {"role": "assistant", "content": "What is your next step?"},
                    {"role": "user", "content": "Order ECG."},
                    {"role": "assistant", "content": "Good, ECG shows ST elevation."}
                ]
                created_at = (now - timedelta(days=N_CASES_PER_USER - j)).isoformat()
                performance_data = {
                    "user_id": user_id,
                    "condition": condition,
                    "result": result_case,
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
                    print(f"Inserted case for {anon_username}: {condition} ({ward})")
                except Exception as e:
                    print(f"Failed to insert case for {anon_username}: {e}")

if __name__ == "__main__":
    main() 