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
from fastapi import FastAPI, Header, HTTPException, Request, status, Depends, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, Dict, Any, Callable, AsyncGenerator
from dotenv import load_dotenv
import uuid
import jwt
import logging
import sys
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import html
import re
from fastapi.routing import APIRoute
import random
import random_username
from random_username.generate import generate_username
import requests
import asyncio

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ukmla-api")

# --- Load Environment Variables ---
load_dotenv()

# Required environment variables
required_env_vars = {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    "OPENAI_ASSISTANT_ID": os.environ.get("OPENAI_ASSISTANT_ID"),
    "SUPABASE_URL": os.environ.get("SUPABASE_URL"),
    "SUPABASE_KEY": os.environ.get("SUPABASE_KEY"),
    "JWT_SECRET": os.environ.get("JWT_SECRET")
}

# Check for missing environment variables
missing_vars = [var for var, value in required_env_vars.items() if not value]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Set up clients and configurations
try:
    # OpenAI configuration
    openai.api_key = required_env_vars["OPENAI_API_KEY"]
    ASSISTANT_ID = required_env_vars["OPENAI_ASSISTANT_ID"]
    
    # Supabase configuration
    SUPABASE_URL = required_env_vars["SUPABASE_URL"]
    SUPABASE_KEY = required_env_vars["SUPABASE_KEY"]
    
    # JWT configuration
    JWT_SECRET = required_env_vars["JWT_SECRET"]
    JWT_ALGORITHM = "HS256"
    
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.error(f"Error initializing configuration: {str(e)}")
    raise

# Initialize clients
try:
    client = openai.OpenAI(
        api_key=openai.api_key,
        default_headers={"OpenAI-Beta": "assistants=v2"},
        timeout=30.0,  # Add timeout
        max_retries=3  # Add retries
    )
    logger.info("OpenAI client initialized successfully with Assistants v2")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    raise

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Supabase client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {str(e)}")
    raise

class CORSRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            if request.method == "OPTIONS":
                return Response(
                    status_code=200,
                    headers={
                        "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
                        "Access-Control-Allow-Methods": "*",
                        "Access-Control-Allow-Headers": "*",
                        "Access-Control-Allow-Credentials": "true",
                    },
                )
            return await original_route_handler(request)

        return custom_route_handler

# Initialize FastAPI app with custom route class
app = FastAPI(
    title="UKMLA Case-Based Tutor API",
    description="API for the UKMLA Case-Based Tutor application",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)
app.router.route_class = CORSRoute

# Add security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]  # In production, replace with your domain
)

# Update CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ukmla-case-tutor.framer.app",  # Production frontend
        "http://localhost:3000",                # Local development
        "https://ukmla-case-tutor.framer.website",
        "https://streamlined-style-184093.framer.app"# Framer website
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods including OPTIONS
    allow_headers=[
        "Content-Type",
        "Authorization", 
        "Accept",
        "X-Refresh-Token",
        "Access-Control-Allow-Headers",
        "Access-Control-Allow-Origin",
        "Access-Control-Allow-Methods",
        "Access-Control-Allow-Credentials"
    ],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- JWT Helper Functions ---
def verify_token(token: str) -> Dict[str, Any]:
    """Verify a JWT token and return its payload."""
    try:
        # Only use JWT_SECRET for verification
        if not JWT_SECRET:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="JWT_SECRET not configured"
            )
            
        try:
            decoded_token = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            logger.info("Successfully verified token")
            return decoded_token
        except jwt.ExpiredSignatureError:
            logger.error("Token has expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except Exception as e:
            logger.error(f"Token verification failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
            
    except Exception as e:
        logger.error(f"Unexpected error in token verification: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

def extract_user_id(token: str) -> str:
    """Extract user ID from token without verifying signature."""
    try:
        # Just decode the payload without verifying signature
        payload = jwt.decode(token, options={"verify_signature": False})
        return payload.get("sub")
    except Exception as e:
        logger.error(f"Error extracting user ID from token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token format"
        )

# Case file directory - handle both /data/cases and /data directories
data_cases_dir = Path(__file__).parent / "data" / "cases"
data_dir = Path(__file__).parent / "data"

if data_cases_dir.exists() and data_cases_dir.is_dir():
    CASE_FILES_DIR = data_cases_dir
    logger.info(f"Using case files from {CASE_FILES_DIR}")
elif data_dir.exists() and data_dir.is_dir():
    CASE_FILES_DIR = data_dir
    logger.info(f"Using case files from {CASE_FILES_DIR}")
else:
    logger.warning("Neither data/cases nor data directory found. Defaulting to data/")
    CASE_FILES_DIR = data_dir

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
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    
    @validator('password')
    def password_strength(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one number')
        return v

class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)

class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., min_length=1)

class LogoutRequest(BaseModel):
    token: str = Field(..., min_length=1)

@app.post("/signup", response_model=dict)
@limiter.limit("3/minute")
async def signup(signup_data: SignupRequest, request: Request):
    """Register a new user account."""
    try:
        result = supabase.auth.sign_up({
            "email": signup_data.email,
            "password": signup_data.password
        })
        if result.user:
            return {"success": True, "message": "Account created. Please verify your email."}
        else:
            return JSONResponse(status_code=400, content={"error": result})
    except Exception as e:
        logger.error(f"Signup failed: {str(e)}")
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/login", response_model=dict)
@limiter.limit("5/minute")
async def login(login_data: LoginRequest, request: Request):
    """Authenticate user and return token/session."""
    try:
        result = supabase.auth.sign_in_with_password({
            "email": login_data.email,
            "password": login_data.password
        })
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
        logger.error(f"Login failed: {str(e)}")
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
    condition: str = Field(..., min_length=1, max_length=100)
    
    @validator('condition')
    def validate_condition(cls, v):
        # Add any specific condition validation rules
        if not v.replace(" ", "").isalnum():
            raise ValueError('Condition must contain only letters, numbers, and spaces')
        return v

class ContinueCaseRequest(BaseModel):
    thread_id: str = Field(..., min_length=1)
    user_input: str = Field(..., min_length=1, max_length=1000)
    token: Optional[str] = None
    refresh_token: Optional[str] = None

@app.get("/wards")
def get_wards(authorization: Optional[str] = Header(None)):
    """Return a nested structure of wards and their cases from the data directory."""
    wards = {}
    
    if not CASE_FILES_DIR.exists() or not CASE_FILES_DIR.is_dir():
        return JSONResponse(status_code=500, content={"error": "Data directory not found."})
    
    # Log which directory is being used
    logger.info(f"Reading wards from {CASE_FILES_DIR}")
    
    # Iterate through each ward directory
    for ward_dir in CASE_FILES_DIR.iterdir():
        if ward_dir.is_dir() and not ward_dir.name.startswith('.'):
            ward_name = ward_dir.name.title()  # Normalize ward name for display
            cases = []
            # Iterate through case files in the ward directory
            case_files = list(ward_dir.glob("*.txt"))
            logger.info(f"Found {len(case_files)} case files in {ward_dir}")
            for case_file in case_files:
                case_name = case_file.stem.replace("_", " ").replace("-", " ").title()
                cases.append(case_name)
                logger.debug(f"Added case: {case_name} to ward: {ward_name}")
            if cases:
                wards[ward_name] = cases
    logger.info(f"Returning wards structure: {wards}")
    return {"wards": wards}

def get_next_case_variation(user_id: str, condition: str) -> int:
    """
    Get the next unused case variation number for this user and condition.
    Args:
        user_id (str): The user's unique identifier (UUID).
        condition (str): The case/condition name.
    Returns:
        int: The next unused variation number (starting from 1).
    """
    try:
        # Query completed cases for this user and condition
        result = supabase.table("performance") \
            .select("case_variation") \
            .eq("user_id", user_id) \
            .eq("condition", condition) \
            .execute()
        if not result.data:
            return 1  # First case

        # Get all used variations (ensure int conversion and skip None)
        used_variations = set(
            int(record.get('case_variation', 0))
            for record in result.data
            if record.get('case_variation') is not None
        )

        # Find the next unused variation number
        variation = 1
        while variation in used_variations:
            variation += 1

        return variation
    except Exception as e:
        logger.error(f"Error in get_next_case_variation for user_id={user_id}, condition={condition}: {str(e)}")
        # If there's an error, default to variation 1
        return 1

async def stream_assistant_response(thread_id: str, run_id: str) -> AsyncGenerator[str, None]:
    """Stream assistant responses using SSE."""
    try:
        while True:
            run = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )
            
            if run.status == "completed":
                messages = client.beta.threads.messages.list(thread_id=thread_id)
                assistant_msg = next((m for m in messages.data if m.role == "assistant"), None)
                if assistant_msg:
                    first_text_block = next((c for c in assistant_msg.content if c.type == "text"), None)
                    if first_text_block:
                        response_data = json.dumps({'content': first_text_block.text.value})
                        yield f"data: {response_data}\n\n"
                break
            elif run.status == "failed":
                error_data = json.dumps({'error': f'Run failed: {run.last_error}'})
                yield f"data: {error_data}\n\n"
                break
            elif run.status == "expired":
                error_data = json.dumps({'error': 'Run expired'})
                yield f"data: {error_data}\n\n"
                break
                
            await asyncio.sleep(0.5)
            
    except Exception as e:
        logger.error(f"Error in stream_assistant_response: {str(e)}")
        error_data = json.dumps({'error': str(e)})
        yield f"data: {error_data}\n\n"

@app.post("/start_case")
async def start_case(request: StartCaseRequest, authorization: str = Header(...)):
    """Start a new case with the OpenAI Assistant using streaming."""
    try:
        # Validate token and get user ID
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header"
            )
        
        token = authorization.split(" ")[1]
        user_id = extract_user_id(token)

        logger.info(f"Starting case for user {user_id} with condition {request.condition}")

        # Get case file and ward
        case_file = get_case_file(request.condition)
        if not case_file:
            raise HTTPException(status_code=404, detail=f"Case '{request.condition}' not found.")
        
        ward = get_ward_for_condition(request.condition)

        # Determine case variation
        case_variation = get_next_case_variation(user_id, request.condition)

        # Read case content
        with open(case_file, "r") as f:
            case_content = f.read()

        # Create thread
        thread = client.beta.threads.create(
            metadata={
                "user_id": user_id,
                "condition": request.condition,
                "ward": ward,
                "start_time": datetime.now(timezone.utc).isoformat(),
                "case_variation": str(case_variation)
            }
        )

        # Send initial case prompt
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"""
GOAL: You are an expert Medical professional with decades of teaching experience. Your goal is to provide UK medicine students with realistic patient cases. Start a UKMLA-style case on: {request.condition} (Variation {case_variation}).

CASE CONTENT:
{case_content}

INSTRUCTIONS:
- Present one case based on the case content provided above.
- Do not skip straight to diagnosis or treatment. Walk through it step-by-step.
- Ask what investigations they'd like, then provide results.
- Nudge the student if they struggle. After 2 failed tries, reveal the answer.
- Use bold for emphasis and to enhance engagement.
- After asking the final question and receiving the answer, output exactly:

[CASE COMPLETED]
{{
    "feedback": "Brief feedback on overall performance",
    "score": number from 1-10
}}

The [CASE COMPLETED] marker must be on its own line, followed by the JSON on new lines.
If the user enters 'SPEEDRUN' I'd like you to do the [CASE COMPLETED] output with a random score and mock feedback
---------------
EXAMPLE 1:Acute Heart Failure Management

You: 🧑‍⚕️ A 78-year-old man presents to the emergency department with increasing shortness of breath over the past 2 days. He is orthopnoeic, has bilateral basal crackles on auscultation, and is using accessory muscles to breathe. His oxygen saturation is 92% on room air. Blood pressure is 145/85 mmHg.

Question 1:
What is the first-line treatment for this patient presenting with acute heart failure?

Student: ld 

You: Correct ✅ — IV loop diuretics (e.g. furosemide) are first-line to reduce fluid overload.

🧑‍⚕️ The patient is given IV furosemide. He still appears breathless and is hypoxic with saturations now 91%.

Question 2:
What is the next step to manage his hypoxia?

Student: oxygen

You: Yes ✅ — Oxygen can be given, but it should be titrated to keep sats between 94–98%, as per BTS guidelines.

🧑‍⚕️ You start oxygen and titrate it appropriately. Despite this, the patient continues to struggle with breathing, showing signs of respiratory fatigue. ABG shows type 1 respiratory failure.

Question 3:
What non-invasive ventilatory support might now be indicated?

Student: cpap

You: Exactly ✅ — CPAP (Continuous Positive Airway Pressure) can support ventilation and reduce preload in acute pulmonary oedema.

🧑‍⚕️ CPAP is started. The patient stabilises over the next hour. However, you are now called to review another patient: a 75-year-old woman with a history of ischaemic cardiomyopathy. She presents with acute breathlessness. BP is 78/55 mmHg, HR 120, and she is clammy and confused.

Question 4:
Given this patient's hypotension, which typical acute heart failure treatments should be used with caution or avoided initially?

Student: diuretics and nitrates

You: 🧑‍⚕️ You suspect cardiogenic shock. Echocardiogram shows severely reduced left ventricular function. There's no tamponade or massive valve lesion.

Question 5:
What medication could be considered to improve cardiac output in this scenario?

Student: dobutamine

You: Spot on ✅ — Dobutamine, an inotropic agent, can support cardiac output in cardiogenic shock with severe LV dysfunction.

🧑‍⚕️ Dobutamine is started. Despite this, the patient shows signs of end-organ hypoperfusion — low urine output and rising lactate.

Question 6:
What agent might you now consider adding if the response to inotropes is inadequate?

Student: iabp 

You: You're thinking ahead — good! ✅ Mechanical circulatory support like IABP can be considered, but before that:

The next step is to add a vasopressor such as norepinephrine, especially if there's ongoing hypotension and end-organ hypoperfusion despite inotropes.

Now let's wrap up:
You've managed two contrasting presentations of acute heart failure — one with preserved blood pressure and pulmonary oedema, the other with cardiogenic shock.

Key takeaways:
IV loop diuretics = mainstay for congestion
Oxygen = titrate to 94–98%, escalate to CPAP if needed
Avoid hypotension-aggravating treatments in shock
Use inotropes (e.g. dobutamine) for low-output states
Add vasopressors (e.g. norepinephrine) if perfusion remains poor
Consider mechanical support like IABP if refractory
---------------
EXAMPLE 2: Mitral stenosis.

You: 🧓 A 58-year-old woman presents with progressive shortness of breath on exertion and a dry cough. She says she struggles going up stairs now and has had two brief episodes of coughing up blood over the past month.

On examination:

She's in atrial fibrillation

There's a malar flush

You hear a low-pitched diastolic murmur, loudest at the apex, especially in expiration, with an opening snap

Her pulse is irregularly irregular and low in volume

Question 1:
Given her symptoms and findings, what is the most likely diagnosis?

Student: mitral stenosis 

You: ✅ Correct — this is classic mitral stenosis.

Next question:
What is the underlying most common cause of mitral stenosis worldwide?

Student: rheumatic fever 

You: ✅ Spot on — rheumatic fever is overwhelmingly the most common cause.

Next question:
She's in atrial fibrillation. What important long-term medication should you consider prescribing to reduce stroke risk?

Student: doac/warfarin

You: 🟡 Almost — you're right that anticoagulation is needed, but:

In moderate to severe mitral stenosis, warfarin is still the recommended anticoagulant — not DOACs, which are not yet widely accepted in this context.

Next question:
She's symptomatic. What would be the first-line interventional procedure for a suitable valve (i.e. non-calcified, mobile leaflets, no LA thrombus)?

Student: baloon valvotomy

You: ✅ Correct — percutaneous mitral balloon valvotomy is first-line in suitable symptomatic patients.

Next question:
What echocardiographic mitral valve area defines "severe" or "tight" mitral stenosis?

Student: <1cm

You: 🟡 Close — let's tighten that up a bit.

The specific signs of left atrial enlargement on chest X-ray include:

Splaying of the carina (widened angle between main bronchi)

Double right heart border

Elevation of the left main bronchus

All of these point toward a dilated left atrium.

✅ Well done — you've worked through a full case of mitral stenosis, from presentation to management!

Would you like to go through another case?
---------------
EXAMPLE 3: Cardiac tamponade

You: Let's work through a case to help you learn cardiac tamponade clearly and practically.

A 58-year-old man is rushed into the emergency department following a stab wound to the chest. He is hypotensive (BP 84/60 mmHg), tachycardic, and appears very breathless. On exam, his JVP is markedly elevated and heart sounds are distant.

Question 1:
What classic triad are you thinking of, and what diagnosis does this suggest?

Student: beck 
cardiac tamponade

You: Correct! You're thinking of Beck's triad — hypotension, raised JVP, and muffled heart sounds — which is classic for cardiac tamponade.

Next part of the case:
His ECG shows a sinus tachycardia with alternating QRS amplitudes beat to beat.

Question 2:
What is this ECG finding called, and why does it occur in tamponade?

Student: occurs due to swinging motion of heart. called ecg alternans

You: Perfect — it's called electrical alternans, and it occurs due to the swinging motion of the heart within a fluid-filled pericardial sac.

Next:
You examine his neck veins and notice no Y descent on the JVP waveform.

Question 3:
Why is the Y descent absent in tamponade?

Student: cos its not cp

You: You're on the right track — it's not constrictive pericarditis (CP), where Y descent is present.

But to be precise:
In tamponade, the Y descent is absent because the right ventricle can't fill properly — the high pericardial pressure restricts diastolic filling, so there's no rapid fall in venous pressure.

Next step:
This man is in shock. Fluids haven't helped.

Question 4:
What is the definitive management for cardiac tamponade?

Student: pericardiocentesis

You: Exactly — urgent pericardiocentesis is the definitive management to relieve the pressure on the heart.

Final check:
Just to tie it all together — if this were constrictive pericarditis instead, name one key CXR finding you'd expect.

Student: calcification

You: Spot on — pericardial calcification on CXR suggests constrictive pericarditis.

✅ Case complete!
You've just worked through the recognition, investigation, and management of cardiac tamponade, and contrasted it with constrictive pericarditis.
"""
        )

        # Create run
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID
        )

        # Return streaming response
        return StreamingResponse(
            stream_assistant_response(thread.id, run.id),
            media_type="text/event-stream",
            headers={
                "X-Thread-Id": thread.id,
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Content-Type": "text/event-stream"
            }
        )


    except Exception as e:
        logger.error(f"❌ start_case failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent XSS and injection attacks."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Escape HTML special characters
    text = html.escape(text)
    # Remove any remaining potentially dangerous characters
    text = re.sub(r'[^\w\s\-.,!?]', '', text)
    return text.strip()

async def stream_continue_case_response(thread_id: str, run_id: str) -> AsyncGenerator[str, None]:
    """Stream assistant responses for continue_case using SSE."""
    try:
        while True:
            run = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )
            
            if run.status == "completed":
                messages = client.beta.threads.messages.list(thread_id=thread_id)
                latest_message = messages.data[0].content[0].text.value
                
                # Check for case completion
                is_completed = "CASE COMPLETED" in latest_message
                feedback = None
                score = None
                
                if is_completed:
                    try:
                        completion_index = latest_message.find("[CASE COMPLETED]")
                        json_text = latest_message[completion_index + len("[CASE COMPLETED]"):].strip()
                        feedback_json = json.loads(json_text)
                        feedback = feedback_json.get("feedback")
                        score = feedback_json.get("score")
                    except Exception as e:
                        logger.error(f"Error processing completion data: {str(e)}")
                
                response_data = json.dumps({
                    'content': latest_message,
                    'is_completed': is_completed,
                    'feedback': feedback,
                    'score': score
                })
                yield f"data: {response_data}\n\n"
                break
            elif run.status == "failed":
                error_data = json.dumps({'error': f'Run failed: {run.last_error}'})
                yield f"data: {error_data}\n\n"
                break
            elif run.status == "expired":
                error_data = json.dumps({'error': 'Run expired'})
                yield f"data: {error_data}\n\n"
                break
                
            await asyncio.sleep(0.5)
            
    except Exception as e:
        logger.error(f"Error in stream_continue_case_response: {str(e)}")
        error_data = json.dumps({'error': str(e)})
        yield f"data: {error_data}\n\n"

@app.post("/continue_case")
async def continue_case(request: ContinueCaseRequest, authorization: str = Header(...)):
    """Continue an existing case with the OpenAI Assistant using streaming."""
    try:
        # Sanitize user input
        sanitized_input = sanitize_input(request.user_input)
        
        # Validate token and get user ID
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header"
            )
        
        token = authorization.split(" ", 1)[1]
        user_id = extract_user_id(token)
        
        logger.info(f"Continuing case for user {user_id} in thread {request.thread_id}")
        
        # Send user message
        try:
            client.beta.threads.messages.create(
                thread_id=request.thread_id,
                role="user",
                content=sanitized_input
            )
            logger.info(f"Sent user message to thread {request.thread_id}")
        except Exception as e:
            error_msg = f"Error sending user message: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
            
        # Create run
        try:
            run = client.beta.threads.runs.create(
                thread_id=request.thread_id,
                assistant_id=ASSISTANT_ID
            )
            
            # Return streaming response
            return StreamingResponse(
                stream_continue_case_response(request.thread_id, run.id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "Content-Type": "text/event-stream"
                }
            )
                
        except Exception as e:
            error_msg = f"Error creating/retrieving run: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
            
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

# --- PERFORMANCE & GAMIFICATION ENDPOINTS ---

class SavePerformanceRequest(BaseModel):
    thread_id: str = Field(..., min_length=1)
    score: int = Field(..., ge=1, le=10)
    feedback: str = Field(..., min_length=1, max_length=1000)
    token: Optional[str] = None
    refresh_token: Optional[str] = None

@app.post("/save_performance")
async def save_performance(request: SavePerformanceRequest, authorization: Optional[str] = Header(None), x_refresh_token: Optional[str] = Header(None)):
    """Save the user's performance after case completion. Returns success and badge info."""
    # Get tokens from either request body or headers
    token = request.token if request.token else (authorization.split(" ", 1)[1] if authorization and authorization.startswith("Bearer ") else None)
    refresh_token = request.refresh_token if request.refresh_token else x_refresh_token

    logger.info(f"/save_performance called with: thread_id={request.thread_id}, score={request.score}, feedback={request.feedback}")
    logger.info(f"Headers: Authorization={authorization}, X-Refresh-Token={x_refresh_token}")
    logger.info(f"Token used: {token}, Refresh token used: {refresh_token}")

    if not token or not refresh_token:
        logger.error("Missing token or refresh token")
        return JSONResponse(status_code=401, content={"error": "Missing token or refresh token"})

    try:
        # Set the session with both tokens for RLS
        supabase.auth.set_session(token, refresh_token)
        logger.info("Supabase session set successfully")
        
        # Get user ID from token
        user_id = extract_user_id(token)
        logger.info(f"Extracted user_id: {user_id}")
        if not user_id:
            logger.error("Invalid token: no user_id")
            return JSONResponse(status_code=401, content={"error": "Invalid token: no user_id"})

        # Get thread metadata to get condition and ward
        thread = client.beta.threads.retrieve(thread_id=request.thread_id)
        metadata = thread.metadata
        condition = metadata.get("condition")
        case_variation = metadata.get("case_variation")
        ward = metadata.get("ward")
        logger.info(f"Thread metadata: condition={condition}, case_variation={case_variation}, ward={ward}")

        if not all([condition, case_variation, ward]):
            logger.error("Missing required metadata in thread")
            return JSONResponse(
                status_code=400,
                content={"error": "Missing required metadata in thread"}
            )

        # Validate score and feedback
        if not 1 <= request.score <= 10:
            logger.error(f"Invalid score: {request.score}")
            return JSONResponse(
                status_code=400,
                content={"error": "Score must be between 1 and 10"}
            )
        if not request.feedback:
            logger.error("Feedback is required but missing")
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
                logger.info(f"Attempting to insert performance_data: {performance_data}")
                result = supabase.table("performance").insert(performance_data).execute()
                logger.info(f"Supabase insert result: {result}")
                if not result.data:
                    logger.error("Failed to save performance data: No data returned from insert")
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
                    logger.info(f"Success count for badge eligibility: {success_count}")
                    # Check if badge already exists
                    badge_result = supabase.table("badges") \
                        .select("id") \
                        .eq("user_id", user_id) \
                        .eq("ward", ward) \
                        .execute()
                    has_badge = bool(badge_result.data)
                    logger.info(f"Badge already exists: {has_badge}")
                    # Grant badge if eligible and not already granted
                    if success_count >= 5 and not has_badge:
                        badge_name = f"{ward} Badge"
                        supabase.table("badges").insert({
                            "user_id": user_id,
                            "ward": ward,
                            "badge_name": badge_name
                        }).execute()
                        badge_awarded = badge_name
                        logger.info(f"Badge awarded: {badge_awarded}")
                except Exception as e:
                    logger.error(f"Error checking badge eligibility: {str(e)}")
                    badge_awarded = None
                return {
                    "success": True,
                    "badge_awarded": badge_awarded
                }
            except Exception as e:
                logger.error(f"Error in Supabase insert attempt {attempt+1}: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)  # Wait before retry
                continue
    except Exception as e:
        logger.error(f"Error in save_performance: {str(e)}")
        logger.error(traceback.format_exc())
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
        user_id = extract_user_id(token)
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
        user_id = extract_user_id(token)
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

def get_case_file(condition_name: str) -> Optional[Path]:
    """Dynamically find the case file path for a given condition name in the data directory."""
    for ward_dir in CASE_FILES_DIR.iterdir():
        if ward_dir.is_dir() and not ward_dir.name.startswith('.'):
            for case_file in ward_dir.glob("*.txt"):
                # Normalize the case name for comparison
                case_name = case_file.stem.replace("_", " ").replace("-", " ").title()
                if case_name == condition_name:
                    return case_file
    return None

def get_ward_for_condition(condition_name: str) -> str:
    """Dynamically find the ward name for a given condition name in the data directory."""
    for ward_dir in CASE_FILES_DIR.iterdir():
        if ward_dir.is_dir() and not ward_dir.name.startswith('.'):
            for case_file in ward_dir.glob("*.txt"):
                case_name = case_file.stem.replace("_", " ").replace("-", " ").title()
                if case_name == condition_name:
                    return ward_dir.name.title()
    return "Unknown"

class OnboardingRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    med_school: str = Field(..., min_length=1, max_length=100)
    year_group: str = Field(..., min_length=1, max_length=10)

@app.post("/onboarding", response_model=dict)
async def onboarding(request: OnboardingRequest, authorization: str = Header(...)):
    """
    Onboard a new user: collect metadata and generate a unique anonymous username.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")

    token = authorization.split(" ", 1)[1]

    try:
        user_id = extract_user_id(token)
        import jwt
        payload = jwt.decode(token, options={"verify_signature": False})
        logger.info(f"Extracted user_id: {user_id}")
        logger.info(f"JWT sub: {payload.get('sub')}")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token.")

    # 2. Check if already onboarded (using Supabase REST API)
    SUPABASE_URL = os.environ["SUPABASE_URL"]
    SUPABASE_KEY = os.environ["SUPABASE_KEY"]
    headers = {
        "Authorization": f"Bearer {token}",
        "apikey": SUPABASE_KEY,
        "Content-Type": "application/json"
    }
    check_resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/user_metadata?user_id=eq.{user_id}",
        headers=headers
    )
    if check_resp.status_code == 200 and check_resp.json():
        raise HTTPException(status_code=409, detail="User already onboarded.")
    elif check_resp.status_code not in (200, 206):
        raise HTTPException(status_code=check_resp.status_code, detail=check_resp.text)

    # 3. Generate anon username
    max_retries = 5
    anon_username = None
    for _ in range(max_retries):
        base_username = generate_username(1)[0]
        suffix = str(random.randint(100, 999))
        candidate = f"med{base_username.lower()}{suffix}"
        check = requests.get(
            f"{SUPABASE_URL}/rest/v1/user_metadata?anon_username=eq.{candidate}",
            headers=headers
        )
        if check.status_code == 200 and not check.json():
            anon_username = candidate
            break
    if not anon_username:
        raise HTTPException(status_code=500, detail="Failed to generate unique anonymous username.")

    # 4. Insert record (using Supabase REST API)
    insert_data = {
        "user_id": user_id,
        "name": request.name,
        "med_school": request.med_school,
        "year_group": request.year_group,
        "anon_username": anon_username
    }
    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/user_metadata",
        headers=headers,
        json=insert_data
    )
    if resp.status_code not in (200, 201):
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return {"anon_username": anon_username}

@app.get("/user_metadata/me", response_model=dict)
async def get_user_metadata_me(authorization: str = Header(...)):
    """
    Retrieve onboarding metadata for the authenticated user.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")

    token = authorization.split(" ", 1)[1]

    try:
        user_id = extract_user_id(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token.")

    SUPABASE_URL = os.environ["SUPABASE_URL"]
    SUPABASE_KEY = os.environ["SUPABASE_KEY"]
    headers = {
        "Authorization": f"Bearer {token}",
        "apikey": SUPABASE_KEY,
        "Accept": "application/vnd.pgrst.object+json"
    }

    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/user_metadata?user_id=eq.{user_id}&select=name,med_school,year_group,anon_username",
        headers=headers
    )

    if resp.status_code == 200:
        return resp.json()
    elif resp.status_code == 406:
        # No record found for this user
        raise HTTPException(status_code=404, detail="User metadata not found.")
    else:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
