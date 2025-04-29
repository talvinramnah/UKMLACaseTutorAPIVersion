import openai
import time
import json
import os
from supabase import create_client
from datetime import datetime, timezone
import traceback
from pathlib import Path
from functools import lru_cache
from utils import (
    rate_limit, 
    check_session_expiry, 
    is_chat_ready,
    get_user_state,
    set_user_state,
    get_user_session_id
)
from fastapi import FastAPI, Header, HTTPException, Request, status, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import uuid
import jwt
import logging

# --- CONFIG ---
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY", "")
ASSISTANT_ID = os.environ.get("OPENAI_ASSISTANT_ID", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

app = FastAPI(
    title="UKMLA Case Tutor API",
    description="API for the UKMLA Case-Based Tutor system",
    version="1.0.0"
)

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize OpenAI client
client = openai.OpenAI(api_key=openai.api_key)

# Ward organization with exact file paths
CASES = {
    "Cardiology": [
        {
            "name": "Acute Coronary Syndrome",
            "file": "acute_coronary_syndrome.txt"
        },
        {
            "name": "Adult Advanced Life Support",
            "file": "adult_advanced_life_support.txt"
        },
        {
            "name": "Hypertension Management",
            "file": "hypertension_management.txt"
        }
    ],
    "Respiratory": [
        {
            "name": "Asthma",
            "file": "asthma.txt"
        },
        {
            "name": "Pneumothorax Management",
            "file": "pneumothorax_management.txt"
        },
        {
            "name": "COPD Management",
            "file": "copd_management.txt"
        }
    ],
    "ENT": [
        {
            "name": "Vestibular Neuronitis",
            "file": "vestibular_neuronitis.txt"
        },
        {
            "name": "Rinne's and Weber's Test",
            "file": "rinnes_and_webers_test.txt"
        },
        {
            "name": "Acute Otitis Media",
            "file": "acute_otitis_media.txt"
        }
    ]
}

def get_case_file(condition_name: str) -> Optional[Path]:
    """Get the case file path for a given condition name."""
    for ward_cases in CASES.values():
        for case in ward_cases:
            if case["name"] == condition_name:
                return CASE_FILES_DIR / case["file"]
    return None

def get_ward_for_condition(condition_name: str) -> str:
    """Get the ward name for a given condition."""
    for ward, cases in CASES.items():
        if any(case["name"] == condition_name for case in cases):
            return ward
    return "Unknown"

@app.get("/")
def read_root():
    return JSONResponse(content={"message": "Welcome to the UKMLA Case Tutor FastAPI backend!"})

# Case file directory - using relative path for cloud deployment
CASE_FILES_DIR = Path(__file__).parent / "data" / "cases"

# --- HELPER FUNCTIONS ---
def wait_for_run_completion(thread_id: str, run_id: str, timeout: int = 60):
    """Wait for run completion with exponential backoff."""
    start_time = time.time()
    wait_time = 0.5
    max_wait = 2.0  # Maximum wait between checks
    
    while True:
        if time.time() - start_time > timeout:
            raise TimeoutError("Response timed out")
            
        status = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        ).status

        if status == "completed":
            return True
        if status == "failed":
            raise Exception("Assistant run failed")
        if status == "expired":
            raise Exception("Assistant run expired")
            
        # Exponential backoff
        time.sleep(min(wait_time, max_wait))
        wait_time *= 1.5

@rate_limit(1)
def send_to_assistant(input_text, thread_id):
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=input_text
    )
    
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID
    )
    
    return run.id

# --- AUTHENTICATION ENDPOINTS ---
class SignupRequest(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class RefreshRequest(BaseModel):
    refresh_token: str

class LogoutRequest(BaseModel):
    token: str

@app.post("/signup", response_model=dict)
def signup(request: SignupRequest):
    """Register a new user account."""
    try:
        result = supabase.auth.sign_up({"email": request.email, "password": request.password})
        if result.user:
            return {"success": True, "message": "Account created. Please verify your email."}
        else:
            return JSONResponse(status_code=400, content={"error": result})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/login", response_model=dict)
def login(request: LoginRequest):
    """Authenticate user and return token/session."""
    try:
        result = supabase.auth.sign_in_with_password({"email": request.email, "password": request.password})
        if result.session:
            return {
                "access_token": result.session.access_token,
                "refresh_token": result.session.refresh_token,
                "user": {
                    "id": result.user.id,
                    "email": result.user.email
                }
            }
        else:
            return JSONResponse(status_code=401, content={"error": "Invalid email or password."})
    except Exception as e:
        return JSONResponse(status_code=401, content={"error": str(e)})

@app.post("/refresh", response_model=dict)
async def refresh(request: RefreshRequest):
    """Refresh session/token using the refresh token."""
    try:
        # Use Supabase auth to refresh the session
        result = supabase.auth.refresh_session(request.refresh_token)
        
        if result.session:
            return {
                "access_token": result.session.access_token,
                "refresh_token": result.session.refresh_token,
                "user": {
                    "id": result.user.id,
                    "email": result.user.email
                }
            }
        else:
            return JSONResponse(
                status_code=401,
                content={"error": "Failed to refresh session. Token may be invalid or expired."}
            )
    except Exception as e:
        return JSONResponse(
            status_code=401,
            content={"error": f"Failed to refresh session: {str(e)}"}
        )

@app.post("/logout", response_model=dict)
async def logout(request: LogoutRequest):
    """Invalidate the current session/token."""
    try:
        # Use Supabase auth to sign out
        # Note: Supabase's sign_out() doesn't require parameters
        supabase.auth.sign_out()
        
        # Always return success as the client-side token is already invalidated
        return {
            "success": True,
            "message": "Successfully logged out"
        }
    except Exception as e:
        # Even if there's an error, we can consider it a successful logout
        # since the client-side token is already invalidated
        return {
            "success": True,
            "message": "Successfully logged out",
            "note": f"Server-side cleanup may have failed: {str(e)}"
        }

# --- CASE MANAGEMENT ENDPOINTS ---
class StartCaseRequest(BaseModel):
    condition: str
    token: str

class ContinueCaseRequest(BaseModel):
    thread_id: str
    user_input: str
    token: str
    refresh_token: str

class SavePerformanceRequest(BaseModel):
    thread_id: str
    score: float
    feedback: str
    token: str

@app.get("/wards", response_model=dict)
def get_wards():
    """Get available wards and their cases."""
    return {
        "wards": [
            {
                "name": ward,
                "cases": [case["name"] for case in cases]
            }
            for ward, cases in CASES.items()
        ]
    }

@app.post("/start_case", response_model=dict)
async def start_case(request: StartCaseRequest):
    """Start a new case with the selected condition."""
    try:
        # Validate token and get user ID
        user_id = get_user_session_id(request.token)
        if not user_id:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid or expired token"}
            )

        # Get case file path
        case_file = get_case_file(request.condition)
        if not case_file:
            return JSONResponse(
                status_code=404,
                content={"error": f"Case '{request.condition}' not found."}
            )

        # Read case content
        with open(case_file, "r") as f:
            case_content = f.read()

        # Get ward for the condition
        ward = get_ward_for_condition(request.condition)

        # Create OpenAI thread with metadata
        thread = client.beta.threads.create(
            metadata={
                "user_id": user_id,
                "condition": request.condition,
                "ward": ward,
                "start_time": datetime.now(timezone.utc).isoformat()
            }
        )

        # Create full case prompt
        case_prompt = f"""You are a medical tutor helping a student practice clinical cases. 
        The case is about {request.condition} in the {ward} ward.
        
        Here is the case content:
        {case_content}
        
        Please:
        1. Start by introducing the case and asking what investigation the student would like to start with
        2. Guide the student through the case step by step
        3. Provide feedback on their decisions
        4. Keep responses concise and focused on learning objectives
        5. Format your completion output as:
           COMPLETION:
           Score: [0-100]
           Feedback: [detailed feedback]
        """

        # Send initial case prompt
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=case_prompt
        )

        # Create and wait for run
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID
        )
        wait_for_run_completion(thread.id, run.id)

        # Get assistant's first message
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        first_message = messages.data[0].content[0].text.value

        # Get case variation
        case_variation = get_next_case_variation(user_id, request.condition)

        return {
            "thread_id": thread.id,
            "message": first_message,
            "case_variation": case_variation
        }

    except TimeoutError:
        return JSONResponse(
            status_code=408,
            content={"error": "Request timed out. Please try again."}
        )
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to start case: {str(e)}"}
        )

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_errors.log'),
        logging.StreamHandler()
    ]
)

@app.post("/continue_case", response_model=dict)
async def continue_case(request: ContinueCaseRequest):
    """Continue an existing case with user input."""
    error_context = {
        "timestamp": datetime.now().isoformat(),
        "thread_id": request.thread_id,
        "user_input": request.user_input
    }
    
    try:
        # Set the session with both tokens for RLS
        try:
            supabase.auth.set_session(request.token, request.refresh_token)
            logging.info(f"Session set successfully for thread {request.thread_id}")
        except Exception as e:
            error_msg = f"Failed to set session: {str(e)}"
            logging.error(f"{error_msg} | Context: {error_context}")
            return JSONResponse(
                status_code=401,
                content={"error": error_msg, "details": str(e)}
            )
        
        # Validate token and get user ID
        try:
            user_id = get_user_session_id(request.token)
            if not user_id:
                error_msg = "Invalid or expired token"
                logging.error(f"{error_msg} | Context: {error_context}")
                return JSONResponse(
                    status_code=401,
                    content={"error": error_msg}
                )
            logging.info(f"Token validated for user {user_id}")
        except Exception as e:
            error_msg = f"Token validation failed: {str(e)}"
            logging.error(f"{error_msg} | Context: {error_context}")
            return JSONResponse(
                status_code=401,
                content={"error": error_msg, "details": str(e)}
            )

        # Send user input to assistant
        try:
            run_id = send_to_assistant(request.user_input, request.thread_id)
            logging.info(f"Message sent to assistant, run_id: {run_id}")
        except Exception as e:
            error_msg = f"Failed to send message to assistant: {str(e)}"
            logging.error(f"{error_msg} | Context: {error_context}")
            return JSONResponse(
                status_code=500,
                content={"error": error_msg, "details": str(e)}
            )
        
        # Wait for response
        try:
            wait_for_run_completion(request.thread_id, run_id)
            logging.info(f"Run completed successfully for run_id: {run_id}")
        except TimeoutError:
            error_msg = "Request timed out. Please try again."
            logging.error(f"{error_msg} | Context: {error_context}")
            return JSONResponse(
                status_code=408,
                content={"error": error_msg}
            )
        except Exception as e:
            error_msg = f"Failed to get assistant response: {str(e)}"
            logging.error(f"{error_msg} | Context: {error_context}")
            return JSONResponse(
                status_code=500,
                content={"error": error_msg, "details": str(e)}
            )
        
        # Get latest message
        try:
            messages = client.beta.threads.messages.list(thread_id=request.thread_id)
            latest_message = messages.data[0].content[0].text.value
            logging.info(f"Retrieved latest message for thread {request.thread_id}")
        except Exception as e:
            error_msg = f"Failed to retrieve messages: {str(e)}"
            logging.error(f"{error_msg} | Context: {error_context}")
            return JSONResponse(
                status_code=500,
                content={"error": error_msg, "details": str(e)}
            )
        
        # Check for case completion
        is_completed = "COMPLETION:" in latest_message
        feedback = None
        score = None
        
        if is_completed:
            try:
                # Extract feedback and score
                completion_parts = latest_message.split("COMPLETION:")[1].strip().split("\n")
                score = float(completion_parts[0].split(":")[1].strip())
                feedback = completion_parts[1].split(":")[1].strip()
                logging.info(f"Case completed with score {score} for thread {request.thread_id}")
                
                # Save performance
                await save_performance(
                    SavePerformanceRequest(
                        thread_id=request.thread_id,
                        score=score,
                        feedback=feedback,
                        token=request.token
                    )
                )
            except Exception as e:
                error_msg = f"Failed to process completion: {str(e)}"
                logging.error(f"{error_msg} | Context: {error_context}")
                return JSONResponse(
                    status_code=500,
                    content={"error": error_msg, "details": str(e)}
                )
        
        return {
            "message": latest_message,
            "is_completed": is_completed,
            "feedback": feedback,
            "score": score
        }

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logging.error(f"{error_msg} | Context: {error_context}")
        logging.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": error_msg, "details": str(e), "traceback": traceback.format_exc()}
        )

@app.post("/save_performance", response_model=dict)
async def save_performance(request: SavePerformanceRequest):
    """Save user performance data for a completed case."""
    try:
        # Validate token and get user ID
        user_id = get_user_session_id(request.token)
        if not user_id:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid or expired token"}
            )

        # Get thread metadata
        thread = client.beta.threads.retrieve(request.thread_id)
        condition = thread.metadata.get("condition")
        ward = thread.metadata.get("ward")
        start_time = datetime.fromisoformat(thread.metadata.get("start_time"))
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        # Save to Supabase
        result = supabase.table("case_performance").insert({
            "user_id": user_id,
            "condition": condition,
            "ward": ward,
            "score": request.score,
            "feedback": request.feedback,
            "duration_seconds": duration,
            "completed_at": end_time.isoformat()
        }).execute()

        # Check for badge eligibility
        await check_badge_eligibility(user_id, condition, request.score)

        return {
            "success": True,
            "message": "Performance data saved successfully"
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to save performance: {str(e)}"}
        )

async def check_badge_eligibility(user_id: str, condition: str, score: float):
    """Check if user is eligible for any badges based on performance."""
    try:
        # Get user's performance history for this condition
        result = supabase.table("case_performance").select("*").eq("user_id", user_id).eq("condition", condition).execute()
        performances = result.data

        # Check for perfect score badge
        if score == 100:
            await award_badge(user_id, "perfect_score", condition)

        # Check for consistent performance badge
        if len(performances) >= 3 and all(p["score"] >= 80 for p in performances[-3:]):
            await award_badge(user_id, "consistent_performer", condition)

    except Exception as e:
        print(f"Error checking badge eligibility: {str(e)}")

async def award_badge(user_id: str, badge_type: str, condition: str):
    """Award a badge to a user."""
    try:
        # Check if badge already awarded
        result = supabase.table("badges").select("*").eq("user_id", user_id).eq("badge_type", badge_type).eq("condition", condition).execute()
        if result.data:
            return

        # Award new badge
        supabase.table("badges").insert({
            "user_id": user_id,
            "badge_type": badge_type,
            "condition": condition,
            "awarded_at": datetime.now(timezone.utc).isoformat()
        }).execute()

    except Exception as e:
        print(f"Error awarding badge: {str(e)}")

def get_next_case_variation(user_id: str, condition: str) -> int:
    """Get the next case variation number for a user and condition."""
    try:
        # Get user's case variations
        result = supabase.table("case_variations").select("*").eq("user_id", user_id).eq("condition", condition).execute()
        variations = result.data

        if not variations:
            # First attempt
            supabase.table("case_variations").insert({
                "user_id": user_id,
                "condition": condition,
                "variation": 1
            }).execute()
            return 1

        # Get next variation
        next_variation = variations[0]["variation"] + 1
        supabase.table("case_variations").update({
            "variation": next_variation
        }).eq("user_id", user_id).eq("condition", condition).execute()

        return next_variation

    except Exception as e:
        print(f"Error getting case variation: {str(e)}")
        return 1 