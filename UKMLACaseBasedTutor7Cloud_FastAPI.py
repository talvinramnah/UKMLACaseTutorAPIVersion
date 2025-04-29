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
from jose import jwt
import logging
from fastapi.middleware.cors import CORSMiddleware

# --- CONFIG ---
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY", "")
ASSISTANT_ID = os.environ.get("OPENAI_ASSISTANT_ID", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip().strip('"\'')  # Strip both whitespace and quotes
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip().strip('"\'')  # Strip both whitespace and quotes

# Debug logging for environment variables
print("=== Environment Variables Debug ===")
print(f"OPENAI_API_KEY length: {len(openai.api_key)}")
print(f"ASSISTANT_ID: {ASSISTANT_ID}")
print(f"SUPABASE_URL: '{SUPABASE_URL}'")  # Added quotes to see any hidden characters
print(f"SUPABASE_URL length: {len(SUPABASE_URL)}")
print(f"SUPABASE_KEY length: {len(SUPABASE_KEY)}")
print("==================================")

# Validate environment variables
if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL environment variable is not set")
if not SUPABASE_KEY:
    raise ValueError("SUPABASE_KEY environment variable is not set")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
if not ASSISTANT_ID:
    raise ValueError("OPENAI_ASSISTANT_ID environment variable is not set")

# Validate Supabase URL format
print("\n=== Supabase URL Validation ===")
print(f"URL starts with https://: {SUPABASE_URL.startswith('https://')}")
print(f"URL ends with .supabase.co: {SUPABASE_URL.endswith('.supabase.co')}")
print("===============================")

if not SUPABASE_URL.startswith("https://") or not SUPABASE_URL.endswith(".supabase.co"):
    raise ValueError(f"Invalid Supabase URL format. Must start with 'https://' and end with '.supabase.co'. Current URL: '{SUPABASE_URL}'")

app = FastAPI(
    title="UKMLA Case Tutor API",
    description="API for the UKMLA Case-Based Tutor system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize Supabase client with error handling
try:
    print("\n=== Supabase Client Initialization ===")
    print("Attempting to initialize Supabase client...")
    print(f"Using URL: {SUPABASE_URL}")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Supabase client initialized successfully")
    print("=====================================")
except Exception as e:
    print(f"\n=== Supabase Initialization Error ===")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("=====================================")
    raise ValueError(f"Failed to initialize Supabase client: {str(e)}")

# Initialize OpenAI client with error handling
try:
    print("\n=== OpenAI Client Initialization ===")
    print("Attempting to initialize OpenAI client...")
    client = openai.OpenAI(api_key=openai.api_key)
    print("OpenAI client initialized successfully")
    print("===================================")
except Exception as e:
    print(f"\n=== OpenAI Initialization Error ===")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("===================================")
    raise ValueError(f"Failed to initialize OpenAI client: {str(e)}")

# Add session validation middleware
@app.middleware("http")
async def validate_session(request: Request, call_next):
    print(f"\n=== Session Validation for {request.url.path} ===")
    
    # Skip session validation for public endpoints
    if request.url.path in ["/", "/login", "/signup", "/refresh"]:
        print("Skipping validation for public endpoint")
        return await call_next(request)
        
    # Get tokens from headers
    auth_header = request.headers.get("Authorization")
    refresh_token = request.headers.get("X-Refresh-Token")
    
    print(f"Auth header present: {bool(auth_header)}")
    print(f"Refresh token present: {bool(refresh_token)}")
    
    if not auth_header or not refresh_token:
        print("Missing token or refresh token")
        return JSONResponse(
            status_code=401,
            content={"error": "Missing token or refresh token"}
        )
        
    try:
        # Extract access token
        token = auth_header.split(" ")[1] if auth_header.startswith("Bearer ") else auth_header
        print(f"Token length: {len(token)}")
        print(f"Refresh token length: {len(refresh_token)}")
        
        try:
            # First try to set the session with the current tokens
            print("Setting Supabase session...")
            supabase.auth.set_session(token, refresh_token)
            print("Session set successfully")
        except Exception as e:
            print(f"Failed to set session with current tokens: {str(e)}")
            # If setting session fails, try to refresh the token
            try:
                print("Attempting to refresh session...")
                result = supabase.auth.refresh_session(refresh_token)
                if result.session:
                    print("Session refreshed successfully")
                    # Update the session with new tokens
                    supabase.auth.set_session(
                        result.session.access_token,
                        result.session.refresh_token
                    )
                    # Update the response headers with new tokens
                    response = await call_next(request)
                    response.headers["X-New-Access-Token"] = result.session.access_token
                    response.headers["X-New-Refresh-Token"] = result.session.refresh_token
                    return response
                else:
                    print("No session returned from refresh")
                    return JSONResponse(
                        status_code=401,
                        content={"error": "Failed to refresh session"}
                    )
            except Exception as refresh_error:
                print(f"Failed to refresh session: {str(refresh_error)}")
                return JSONResponse(
                    status_code=401,
                    content={"error": "Session validation failed"}
                )
        
        # Continue with the request
        response = await call_next(request)
        return response
        
    except Exception as e:
        print(f"Session validation error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return JSONResponse(
            status_code=401,
            content={"error": f"Session validation failed: {str(e)}"}
        )

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
    print("\n=== Login Attempt ===")
    print(f"Email: {request.email}")
    
    try:
        # Sign in with password
        print("Attempting Supabase sign in...")
        result = supabase.auth.sign_in_with_password({
            "email": request.email,
            "password": request.password
        })
        
        if not result.session:
            print("No session returned from Supabase")
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid email or password."}
            )
            
        print("Login successful, setting session...")
        # Set the session for RLS
        supabase.auth.set_session(
            result.session.access_token,
            result.session.refresh_token
        )
        print("Session set successfully")
        
        # Ensure we return a valid JSON response
        response_data = {
            "access_token": result.session.access_token,
            "refresh_token": result.session.refresh_token,
            "user": {
                "id": result.user.id,
                "email": result.user.email
            }
        }
        
        print("Returning response with tokens")
        return JSONResponse(
            status_code=200,
            content=response_data
        )
    except Exception as e:
        print(f"Login error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return JSONResponse(
            status_code=401,
            content={"error": str(e)}
        )

@app.post("/refresh", response_model=dict)
async def refresh(request: RefreshRequest):
    """Refresh session/token using the refresh token."""
    print("\n=== Refresh Attempt ===")
    print(f"Refresh token length: {len(request.refresh_token)}")
    
    try:
        # Use Supabase auth to refresh the session
        print("Attempting to refresh session...")
        result = supabase.auth.refresh_session(request.refresh_token)
        
        if not result.session:
            print("No session returned from refresh")
            return JSONResponse(
                status_code=401,
                content={"error": "Failed to refresh session. Token may be invalid or expired."}
            )
            
        print("Refresh successful, setting new session...")
        # Set the new session for RLS
        supabase.auth.set_session(
            result.session.access_token,
            result.session.refresh_token
        )
        print("New session set successfully")
        
        return {
            "access_token": result.session.access_token,
            "refresh_token": result.session.refresh_token,
            "user": {
                "id": result.user.id,
                "email": result.user.email
            }
        }
    except Exception as e:
        print(f"Refresh error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
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

class ContinueCaseRequest(BaseModel):
    thread_id: str
    user_input: str
    token: Optional[str] = None
    refresh_token: Optional[str] = None

@app.get("/wards")
def get_wards(authorization: Optional[str] = Header(None)):
    """Return a list of all cases in data/cases as a flat structure under a generic ward."""
    try:
        cases_dir = CASE_FILES_DIR
        if not cases_dir.exists() or not cases_dir.is_dir():
            return JSONResponse(
                status_code=500,
                content={"error": "Cases directory not found."}
            )
            
        wards = {"All": []}
        for case_file in cases_dir.glob("*.txt"):
            case_name = case_file.stem.replace("_", " ").title()
            wards["All"].append(case_name)
            
        return JSONResponse(
            status_code=200,
            content={"wards": wards}
        )
    except Exception as e:
        print(f"Wards error: {str(e)}")  # Add logging
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/start_case")
def start_case(request: StartCaseRequest, authorization: Optional[str] = Header(None)):
    """Start a new case for the authenticated user. Returns thread_id, first_message, and case_variation."""
    # 1. Validate token and extract user_id
    if not authorization or not authorization.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"error": "Missing or invalid Authorization header."})
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, options={"verify_signature": False})
        user_id = payload.get("sub")
        if not user_id:
            return JSONResponse(status_code=401, content={"error": "Invalid token: no user_id."})
    except Exception as e:
        return JSONResponse(status_code=401, content={"error": f"Invalid token: {str(e)}"})

    # 2. Find the requested case file
    case_file = get_case_file(request.condition)
    if not case_file:
        return JSONResponse(status_code=404, content={"error": f"Case '{request.condition}' not found."})

    try:
        # 3. Read the case content
        with open(case_file, "r") as f:
            case_content = f.read().strip()

        # 4. Get next case variation
        case_variation = get_next_case_variation(request.condition, user_id)

        # 5. Create OpenAI thread with metadata
        thread = client.beta.threads.create(
            metadata={
                "user_id": user_id,
                "condition": request.condition,
                "case_variation": str(case_variation),
                "ward": get_ward_for_condition(request.condition)
            }
        )

        # 6. Create initial message with case content
        initial_prompt = f"""
GOAL: Start a UKMLA-style case on: {request.condition} (Variation {case_variation}).

PERSONA: You are a senior doctor training a medical student through a real-life ward case for the UKMLA.

CASE CONTENT:
{case_content}

INSTRUCTIONS:
- Present one case based on the case content provided above.
- Do not skip straight to diagnosis or treatment. Walk through it step-by-step.
- Ask what investigations they'd like, then provide results.
- Nudge the student if they struggle. After 2 failed tries, reveal the answer.
- Encourage and use emojis + bold to engage.
- After asking the final question and receiving the answer, output exactly:

[CASE COMPLETED]
{{
    "feedback": "Brief feedback on overall performance",
    "score": number from 1-10
}}

The [CASE COMPLETED] marker must be on its own line, followed by the JSON on new lines.
-if the user enters 'SPEEDRUN' I'd like you to do the [CASE COMPLTED] output with a random score and mock feedback
"""

        # 7. Send initial message to thread
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=initial_prompt
        )

        # 8. Create and run the assistant
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID
        )

        # 9. Wait for run completion
        start_time = time.time()
        wait_time = 0.5
        max_wait = 2.0
        
        while True:
            if time.time() - start_time > 60:  # 60 second timeout
                raise TimeoutError("Case initialization timed out")
                
            status = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            ).status

            if status == "completed":
                break
            if status == "failed":
                raise Exception("Assistant run failed")
            if status == "expired":
                raise Exception("Assistant run expired")

            time.sleep(min(wait_time, max_wait))
            wait_time *= 1.5

        # 10. Get the assistant's first message
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        first_message = None
        for msg in messages:
            if msg.role == "assistant":
                first_message = msg.content[0].text.value
                break

        if not first_message:
            raise Exception("No assistant message found")

        return {
            "thread_id": thread.id,
            "first_message": first_message,
            "case_variation": case_variation
        }

    except TimeoutError as e:
        return JSONResponse(status_code=504, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to start case: {str(e)}"})

def get_next_case_variation(condition: str, user_id: str) -> int:
    """Get the next unused case variation number for this user and condition."""
    try:
        # Query completed cases for this user and condition
        result = supabase.table("performance").select("case_variation").eq("user_id", user_id).eq("condition", condition).execute()
        
        if not result.data:
            return 1  # First case
            
        # Get all used variations
        used_variations = set(record.get('case_variation', 0) for record in result.data)
        
        # Find the next unused variation number
        variation = 1
        while variation in used_variations:
            variation += 1
            
        return variation
    except Exception as e:
        # If there's an error, default to variation 1
        return 1

@app.post("/continue_case", response_model=dict)
async def continue_case(request: ContinueCaseRequest, authorization: Optional[str] = Header(None), x_refresh_token: Optional[str] = Header(None)):
    """Continue an existing case with user input."""
    # Get tokens from either request body or headers
    token = request.token if request.token else (authorization.split(" ", 1)[1] if authorization and authorization.startswith("Bearer ") else None)
    refresh_token = request.refresh_token if request.refresh_token else x_refresh_token

    if not token or not refresh_token:
        return JSONResponse(
            status_code=401,
            content={"error": "Missing token or refresh token"}
        )

    try:
        # Set the session with both tokens for RLS
        supabase.auth.set_session(token, refresh_token)
        
        try:
            # Send user input to assistant
            run_id = send_to_assistant(request.user_input, request.thread_id)
            
            # Wait for response
            wait_for_run_completion(request.thread_id, run_id)
            
            # Get latest message
            messages = client.beta.threads.messages.list(thread_id=request.thread_id)
            latest_message = messages.data[0].content[0].text.value
            
            # Check for case completion
            is_completed = "[CASE COMPLETED]" in latest_message
            feedback = None
            score = None
            
            if is_completed:
                try:
                    # Extract JSON after the completion marker
                    completion_index = latest_message.find("[CASE COMPLETED]")
                    json_text = latest_message[completion_index:].strip()
                    json_text = json_text.replace("[CASE COMPLETED]", "").strip()
                    feedback_json = json.loads(json_text)
                    
                    score = int(feedback_json["score"])
                    feedback = feedback_json["feedback"]
                    
                    # Validate score
                    if not 1 <= score <= 10:
                        raise ValueError("Score must be between 1 and 10")
                    
                    # Validate feedback
                    if not feedback or not isinstance(feedback, str):
                        raise ValueError("Feedback must be a non-empty string")
                        
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse completion JSON: {str(e)}")
                    return JSONResponse(
                        status_code=500,
                        content={"error": "Invalid completion format"}
                    )
                except ValueError as e:
                    logging.error(f"Invalid feedback data: {str(e)}")
                    return JSONResponse(
                        status_code=500,
                        content={"error": str(e)}
                    )
            
            return {
                "assistant_reply": latest_message,
                "case_completed": is_completed,
                "feedback": feedback,
                "score": score
            }

        except TimeoutError:
            return JSONResponse(
                status_code=408,
                content={"error": "Request timed out. Please try again."}
            )
        except Exception as e:
            logging.error(f"Error in continue_case: {str(e)}")
            logging.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to continue case: {str(e)}"}
            )

    except Exception as e:
        logging.error(f"Authentication error in continue_case: {str(e)}")
        logging.error(traceback.format_exc())
        return JSONResponse(
            status_code=401,
            content={"error": f"Authentication failed: {str(e)}"}
        )

# --- PERFORMANCE & GAMIFICATION ENDPOINTS ---

class SavePerformanceRequest(BaseModel):
    thread_id: str
    score: int
    feedback: str
    token: Optional[str] = None
    refresh_token: Optional[str] = None

@app.post("/save_performance")
async def save_performance(request: SavePerformanceRequest, authorization: Optional[str] = Header(None), x_refresh_token: Optional[str] = Header(None)):
    """Save the user's performance after case completion. Returns success and badge info."""
    # Get tokens from either request body or headers
    token = request.token if request.token else (authorization.split(" ", 1)[1] if authorization and authorization.startswith("Bearer ") else None)
    refresh_token = request.refresh_token if request.refresh_token else x_refresh_token

    if not token or not refresh_token:
        return JSONResponse(status_code=401, content={"error": "Missing token or refresh token"})

    try:
        # Set the session with both tokens for RLS
        supabase.auth.set_session(token, refresh_token)
        
        # Get user ID from token
        payload = jwt.decode(token, options={"verify_signature": False})
        user_id = payload.get("sub")
        if not user_id:
            return JSONResponse(status_code=401, content={"error": "Invalid token: no user_id"})

        # Get thread metadata to get condition and ward
        thread = client.beta.threads.retrieve(thread_id=request.thread_id)
        metadata = thread.metadata
        condition = metadata.get("condition")
        case_variation = metadata.get("case_variation")
        ward = metadata.get("ward")

        if not all([condition, case_variation, ward]):
            return JSONResponse(
                status_code=400,
                content={"error": "Missing required metadata in thread"}
            )

        # Validate score and feedback
        if not 1 <= request.score <= 10:
            return JSONResponse(
                status_code=400,
                content={"error": "Score must be between 1 and 10"}
            )
        if not request.feedback:
            return JSONResponse(
                status_code=400,
                content={"error": "Feedback is required"}
            )

        # Retry logic for Supabase operations
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Save performance data
                performance_data = {
                    "user_id": user_id,
                    "condition": condition,
                    "case_variation": case_variation,
                    "score": request.score,
                    "feedback": request.feedback,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "ward": ward
                }
                
                result = supabase.table("performance").insert(performance_data).execute()
                if not result.data:
                    raise Exception("Failed to save performance data")
                
                # Check for badge eligibility
                badge_awarded = None
                try:
                    # Count successful cases in this ward (score >= 7)
        perf_result = supabase.table("performance") \
            .select("score") \
            .eq("user_id", user_id) \
            .eq("ward", ward) \
            .gte("score", 7) \
            .execute()
                    
        success_count = len(perf_result.data) if perf_result.data else 0

                    # Check if badge already exists
        badge_result = supabase.table("badges") \
            .select("id") \
            .eq("user_id", user_id) \
            .eq("ward", ward) \
            .execute()
                    
        has_badge = bool(badge_result.data)

                    # Grant badge if eligible and not already granted
        if success_count >= 5 and not has_badge:
            badge_name = f"{ward} Badge"
            supabase.table("badges").insert({
                "user_id": user_id,
                "ward": ward,
                "badge_name": badge_name
            }).execute()
                        badge_awarded = badge_name
    except Exception as e:
                    logging.error(f"Error checking badge eligibility: {str(e)}")
                    # Don't fail the whole operation if badge check fails
                    badge_awarded = None
                
                return {
                    "success": True,
                    "badge_awarded": badge_awarded
                }
                
                                except Exception as e:
                if attempt == max_retries - 1:
                    logging.error(f"Failed to save performance after {max_retries} attempts: {str(e)}")
                    raise
                                    time.sleep(1)  # Wait before retry

        except Exception as e:
        logging.error(f"Error in save_performance: {str(e)}")
        logging.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to save performance: {str(e)}"}
        )

@app.get("/badges")
def get_badges(authorization: Optional[str] = Header(None), x_refresh_token: Optional[str] = Header(None)):
    """Get all badges for the authenticated user."""
    # 1. Validate token and extract user_id
    if not authorization or not authorization.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"error": "Missing or invalid Authorization header."})
    if not x_refresh_token:
        return JSONResponse(status_code=401, content={"error": "Missing refresh token header."})
    token = authorization.split(" ", 1)[1]
    supabase.auth.set_session(token, x_refresh_token)  # Set session for RLS
    try:
        payload = jwt.decode(token, options={"verify_signature": False})
        user_id = payload.get("sub")
        if not user_id:
            return JSONResponse(status_code=401, content={"error": "Invalid token: no user_id."})
    except Exception as e:
        return JSONResponse(status_code=401, content={"error": f"Invalid token: {str(e)}"})

    # 2. Query badges table for this user
    try:
        result = supabase.table("badges").select("ward, badge_name, earned_at").eq("user_id", user_id).execute()
        badges = result.data if result.data else []
        return {"badges": badges}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to fetch badges: {str(e)}"})

# --- USER PROGRESS ENDPOINT (OPTIONAL) ---

@app.get("/progress")
def get_progress(authorization: Optional[str] = Header(None), x_refresh_token: Optional[str] = Header(None)):
    """Get the user's progress: completed cases, scores, badges, and statistics."""
    # 1. Validate token and extract user_id
    if not authorization or not authorization.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"error": "Missing or invalid Authorization header."})
    if not x_refresh_token:
        return JSONResponse(status_code=401, content={"error": "Missing refresh token header."})
    token = authorization.split(" ", 1)[1]
    supabase.auth.set_session(token, x_refresh_token)  # Set session for RLS
    try:
        payload = jwt.decode(token, options={"verify_signature": False})
        user_id = payload.get("sub")
        if not user_id:
            return JSONResponse(status_code=401, content={"error": "Invalid token: no user_id."})
    except Exception as e:
        return JSONResponse(status_code=401, content={"error": f"Invalid token: {str(e)}"})

    try:
        # 2. Get completed cases and scores
        perf_result = supabase.table("performance") \
            .select("condition, score, feedback, ward, created_at") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .execute()
        
        completed_cases = perf_result.data if perf_result.data else []
        
        # 3. Calculate statistics
        total_cases = len(completed_cases)
        total_score = sum(case["score"] for case in completed_cases)
        avg_score = total_score / total_cases if total_cases > 0 else 0
        successful_cases = len([case for case in completed_cases if case["score"] >= 7])
        success_rate = (successful_cases / total_cases * 100) if total_cases > 0 else 0
        
        # 4. Get badges
        badges_result = supabase.table("badges") \
            .select("ward, badge_name, earned_at") \
            .eq("user_id", user_id) \
            .execute()
        
        badges = badges_result.data if badges_result.data else []
        
        # 5. Get ward-specific statistics
        ward_stats = {}
        for case in completed_cases:
            ward = case["ward"]
            if ward not in ward_stats:
                ward_stats[ward] = {
                    "total_cases": 0,
                    "total_score": 0,
                    "successful_cases": 0
                }
            ward_stats[ward]["total_cases"] += 1
            ward_stats[ward]["total_score"] += case["score"]
            if case["score"] >= 7:
                ward_stats[ward]["successful_cases"] += 1
        
        # Calculate ward averages
        for ward in ward_stats:
            stats = ward_stats[ward]
            stats["avg_score"] = stats["total_score"] / stats["total_cases"]
            stats["success_rate"] = (stats["successful_cases"] / stats["total_cases"] * 100)
        
        return {
            "overall": {
                "total_cases": total_cases,
                "total_score": total_score,
                "average_score": round(avg_score, 2),
                "successful_cases": successful_cases,
                "success_rate": round(success_rate, 2),
                "total_badges": len(badges)
            },
            "ward_stats": ward_stats,
            "recent_cases": completed_cases[:5],  # Last 5 cases
            "badges": badges
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to fetch progress: {str(e)}"})

# --- (Other FastAPI endpoints and backend logic will go here) ---
