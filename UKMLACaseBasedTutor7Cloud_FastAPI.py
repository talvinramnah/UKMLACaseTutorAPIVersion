import openai
import time
import json
import os
from supabase import create_client
from datetime import datetime, timezone, timedelta
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
from fastapi import FastAPI, Header, HTTPException, Request, status, Depends, Response, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, Dict, Any, Callable, AsyncGenerator, List
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
import random
import string
from english_words import english_words_set
import requests
import asyncio
import hashlib
from dateutil.relativedelta import relativedelta

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
        timeout=60.0,  # Increased timeout for streaming runs
        max_retries=2
    )
    logger.info("OpenAI client initialized successfully with Assistants v2 (optimized timeouts)")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    raise

# --- CONNECTION POOLING FOR SUPABASE ---
def create_optimized_supabase_client():
    """Create Supabase client with optimized settings"""
    # Simple client creation without custom session for compatibility
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# Singleton pattern for Supabase client
_supabase_client = None

def get_supabase_client():
    """Get singleton Supabase client"""
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_optimized_supabase_client()
        logger.info("Supabase client initialized successfully")
    return _supabase_client

try:
    supabase = get_supabase_client()
    logger.info("Supabase client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {str(e)}")
    raise

# --- RESPONSE CACHING SYSTEM ---
import hashlib
from typing import Optional as OptionalType

# In-memory cache for responses (use Redis in production for scaling)
RESPONSE_CACHE = {}
CACHE_MAX_SIZE = 100  # Limit cache size to prevent memory issues
CACHE_TTL = 3600  # Cache for 1 hour (in seconds)

def generate_cache_key(condition: str, case_content_hash: str, user_variation: int = None) -> str:
    """Generate a unique cache key for case starts"""
    key_components = [condition, case_content_hash]
    if user_variation is not None:
        key_components.append(str(user_variation))
    
    key_string = "_".join(key_components)
    return hashlib.md5(key_string.encode()).hexdigest()

def get_cached_response(cache_key: str) -> OptionalType[Dict[str, Any]]:
    """Retrieve cached response if it exists and hasn't expired"""
    if cache_key not in RESPONSE_CACHE:
        return None
    
    cached_item = RESPONSE_CACHE[cache_key]
    current_time = time.time()
    
    # Check if cache has expired
    if current_time - cached_item["timestamp"] > CACHE_TTL:
        del RESPONSE_CACHE[cache_key]
        logger.info(f"Cache expired for key: {cache_key[:8]}...")
        return None
    
    logger.info(f"Cache hit for key: {cache_key[:8]}...")
    return cached_item["response"]

def set_cached_response(cache_key: str, response: Dict[str, Any]) -> None:
    """Cache a response with timestamp"""
    # Implement simple LRU by removing oldest entries if cache is full
    if len(RESPONSE_CACHE) >= CACHE_MAX_SIZE:
        # Remove the oldest entry
        oldest_key = min(RESPONSE_CACHE.keys(), 
                        key=lambda k: RESPONSE_CACHE[k]["timestamp"])
        del RESPONSE_CACHE[oldest_key]
        logger.info(f"Cache full, removed oldest entry: {oldest_key[:8]}...")
    
    RESPONSE_CACHE[cache_key] = {
        "response": response,
        "timestamp": time.time()
    }
    logger.info(f"Cached response for key: {cache_key[:8]}...")

def hash_content(content: str) -> str:
    """Generate hash for content to use in cache keys"""
    return hashlib.md5(content.encode()).hexdigest()

# --- END CACHING SYSTEM ---

# --- FILE I/O CACHING SYSTEM ---
FILE_CACHE = {}
FILE_CACHE_MAX_SIZE = 50  # Cache up to 50 files
FILE_CACHE_TTL = 1800     # Cache files for 30 minutes

def get_cached_file_content(file_path: Path) -> OptionalType[str]:
    """Get cached file content if available and not expired"""
    cache_key = str(file_path)
    
    if cache_key not in FILE_CACHE:
        return None
    
    cached_item = FILE_CACHE[cache_key]
    current_time = time.time()
    
    # Check if cache has expired
    if current_time - cached_item["timestamp"] > FILE_CACHE_TTL:
        del FILE_CACHE[cache_key]
        logger.info(f"File cache expired for: {file_path.name}")
        return None
    
    logger.info(f"File cache hit for: {file_path.name}")
    return cached_item["content"]

def set_cached_file_content(file_path: Path, content: str) -> None:
    """Cache file content with timestamp"""
    # Implement simple LRU by removing oldest entries if cache is full
    if len(FILE_CACHE) >= FILE_CACHE_MAX_SIZE:
        # Remove the oldest entry
        oldest_key = min(FILE_CACHE.keys(), 
                        key=lambda k: FILE_CACHE[k]["timestamp"])
        del FILE_CACHE[oldest_key]
        logger.info(f"File cache full, removed oldest entry")
    
    cache_key = str(file_path)
    FILE_CACHE[cache_key] = {
        "content": content,
        "timestamp": time.time()
    }
    logger.info(f"Cached file content for: {file_path.name}")

def read_case_file_cached(file_path: Path) -> str:
    """Read case file with caching to avoid repeated disk I/O"""
    # Check cache first
    cached_content = get_cached_file_content(file_path)
    if cached_content is not None:
        return cached_content
    
    # Read from disk if not cached
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Cache the content
        set_cached_file_content(file_path, content)
        logger.info(f"Read and cached file: {file_path.name}")
        return content
    
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise

# --- END FILE CACHING SYSTEM ---

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
        "https://streamlined-style-184093.framer.app",
        "https://ukmla-frontend-878aksx2x-talvinramnahs-projects.vercel.app",
        "https://ukmla-frontend.vercel.app",
        "https://bleep64.com",
        "https://www.bleep64.com"
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

# --- UTILITY: Enumerate all wards and conditions ---
def enumerate_wards_and_conditions() -> dict:
    """
    Returns a dictionary of all wards and their conditions from the data directory.
    Format: { ward: [condition1, condition2, ...], ... }
    Ward and condition names are formatted for display and routing.
    """
    wards_conditions = {}
    for ward_dir in CASE_FILES_DIR.iterdir():
        if ward_dir.is_dir() and not ward_dir.name.startswith('.'):
            ward_name = ward_dir.name.replace('_', ' ').replace('-', ' ').title()
            conditions = []
            for case_file in ward_dir.glob("*.txt"):
                # Exclude summary/all_conditions files
                if case_file.stem.lower() in ("all_conditions", "summary"): continue
                condition_name = case_file.stem.replace('_', ' ').replace('-', ' ').title()
                conditions.append(condition_name)
            if conditions:
                wards_conditions[ward_name] = sorted(conditions)
    return wards_conditions

# --- HELPER FUNCTIONS ---
def wait_for_run_completion(thread_id: str, run_id: str, timeout: int = 60):
    """Wait for run completion with optimized exponential backoff."""
    start_time = time.time()
    wait_time = 0.1  # Start with 100ms for faster initial responses
    max_wait = 1.0   # Reduced max wait from 2.0s to 1.0s
    
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
            
        # Optimized exponential backoff: faster initial checks, capped at 1s
        time.sleep(min(wait_time, max_wait))
        wait_time = min(wait_time * 1.5, max_wait)  # Ensure we don't exceed max_wait

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
    password: str = Field(..., min_length=8, max_length=100)  # Only length validation

class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)

class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., min_length=1)

class LogoutRequest(BaseModel):
    token: str = Field(..., min_length=1)

# NOTE: Rate limit relaxed for bulk user creation/testing. Restore to '3/minute' after testing.
@app.post("/signup", response_model=dict)
@limiter.limit("3/minute")  # Standard rate limit for signup
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
    case_focus: Optional[str] = Field(None, description="Focus of the case: investigation, management, or both")
    
    @validator('condition')
    def validate_condition(cls, v):
        if not v.replace(" ", "").isalnum():
            raise ValueError('Condition must contain only letters, numbers, and spaces')
        return v
    @validator('case_focus')
    def validate_case_focus(cls, v):
        if v is not None and v not in ("investigation", "management", "both"):
            raise ValueError('case_focus must be one of: investigation, management, both')
        return v

class ContinueCaseRequest(BaseModel):
    thread_id: str = Field(..., min_length=1)
    user_input: str = Field(..., min_length=1, max_length=1000)
    token: Optional[str] = None
    refresh_token: Optional[str] = None

# --- UX NAVIGATION MODELS ---

class NewCaseSameConditionRequest(BaseModel):
    thread_id: str = Field(..., min_length=1)

class ThreadInfoResponse(BaseModel):
    thread_id: str
    condition: str
    ward: str
    is_completed: bool
    user_id: str
    start_time: str

class SessionStateResponse(BaseModel):
    active_threads: List[Dict[str, Any]]
    recent_cases: List[Dict[str, Any]]
    user_progress: Dict[str, Any]

@app.get("/wards")
def get_wards(authorization: Optional[str] = Header(None)):
    """Return a nested structure of wards and their cases from the data directory."""
    wards = {}
    
    if not CASE_FILES_DIR.exists() or not CASE_FILES_DIR.is_dir():
        return JSONResponse(status_code=500, content={"error": "Data directory not found."})
    
    # Log which directory is being used
    logger.info(f"Reading wards from {CASE_FILES_DIR}")
    
    # Iterate through each ward directory (sorted alphabetically)
    for ward_dir in sorted(CASE_FILES_DIR.iterdir()):
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
                # Sort cases alphabetically
                wards[ward_name] = sorted(cases)
    logger.info(f"Returning wards structure: {wards}")
    return {"wards": wards}

@app.post("/start_case")
async def start_case(request: StartCaseRequest, authorization: str = Header(...)):
    """Start a new case with the OpenAI Assistant using streaming. Accepts optional case_focus parameter."""
    try:
        # Validate token and get user ID
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header"
            )
        token = authorization.split(" ")[1]
        user_id = extract_user_id(token)
        logger.info(f"Starting case for user {user_id} with condition {request.condition} and case_focus {request.case_focus}")
        # Get case file and ward
        case_file = get_case_file(request.condition)
        if not case_file:
            raise HTTPException(status_code=404, detail=f"Case '{request.condition}' not found.")
        ward = get_ward_for_condition(request.condition)
        # Read case content using cached file reading
        case_content = read_case_file_cached(case_file)
        # --- Build case focus instruction ---
        focus_instruction = ""
        if request.case_focus == "investigation":
            focus_instruction = "For this case, focus ONLY on investigation. Do not ask or discuss management. Once the investigation is complete do not ask any questions about management. Move to the case completion steps immediately after the investigation is complete."
        elif request.case_focus == "management":
            focus_instruction = "For this case, focus ONLY on management. Do not ask or discuss investigation. All relevant investigations for this specific patient should have occured previously. And hence results should be given to user in questions"
        # --- Compose system prompt ---
        system_prompt = f"""
IMPORTANT: 
- You have access to a GOOD example case (acne vulgaris) in the vector store (file_id: file-3MvudA21kdrQiXU2yKgnaV, metadata: {{condition: 'acne vulgaris', quality: 'good'}}).
- You have access to a GOOD example case (Acute Otits Media) in the vector store (file_id: file-2tP7FDWCpZhALrJJ21hVKt, metadata: {{condition: 'Acute Otits Media', quality: 'good'}}).
- You have access to a GOOD example case for **investigation only** (pneumothorax) in the vector store (file_id: file-AyEyyZDXJBzxQ1QenveJqN, metadata: {{condition: 'pneumothorax', quality: 'good', focus: 'investigation'}}).

When building a case, follow the structure and style used in the attached GOOD example (acne vulgaris case). Refer to its tone, flow, and level of detail to maintain quality. Ignore any patterns from poorly structured transcripts.

QUALITY GUARD: Before presenting the case to the student, compare your structure and logic to the GOOD acne case. If your structure lacks clinical clarity, stepwise progression, or specificity, revise it to match that format before outputting.

GOAL: You are an expert UK-based medical educator with decades of experience. Begin a realistic, step-by-step UKMLA-style clinical case for the condition: **{request.condition}**.

CASE CONTENT (for internal guidance only – do not reveal directly):
{case_content}

{focus_instruction}

INSTRUCTIONS:

PATIENT INTRODUCTION:
Begin with a detailed randomised fictional patient profile using the following structure:

1.  
**Name**, **Age**, **NHS number**, **Date of birth**, **Ethnicity**  
→ Age and date of birth must corrospond i.e. if the age is 20, the date of birth must be 20 years ago, it's currently June 2025. Name and ethnicity must be consistent and realistic.

2.  
**Presenting complaint** — one clear sentence using SOCRATES where relevant (e.g. "Sudden onset central chest pain radiating to the left arm.")  
**History of presenting complaint** — concise clinical story that fits with {request.condition}  
**Medical history** — only relevant past conditions  
**Drug history** — include both prescribed and over-the-counter medications  
**Family history** — if relevant  

3.  
**Ideas**, **Concerns**, **Expectations** — short but believable reflections from the patient

CASE DELIVERY:
- Ask questions **one at a time** to guide the student through the case.
- Ensure proper spacing between questions and answers. Ensure proper formatting i.e. bold titles and headings and callouts for important information.
- Do **not** begin with a question about the diagnosis or generic ambiguity. Start with a specific, relevant clinical question (e.g. "What is the first investigation?").
- Do **not** ask the user the question "what would you want to ask next", or "what would you like to do next", or "what would you like to do now".
- Focus only on the scope of **{request.condition}**. Do **not** introduce related conditions or distractors unless directly high-yield for UKMLA.
- Avoid any multiple choice or list format. Ask clear, open-ended questions that are specific and direct.
- Encourage the student gently. Rephrase if they get it wrong. After 2 failed attempts, provide the correct answer and move forward.
- If a nonsense answer is given (e.g. "carrot", "yes", "no", "I don't know", or any slurs), refuse it politely and re-ask the question clearly.
- Use **bold** to highlight key terms, test results, and medications.
- Build logically: each question should follow from the last (e.g. results → management → monitoring).
- Do **not** tell the student what medication or treatment to give. Always ask first: "What medication would you give here?"
- Avoid over-teaching. Use short summaries and nudges rather than long paragraphs.
- Prioritise **high-yield UKMLA content**. Skip niche or ultra-rare detail.
- Review chat history before asking a new question.
- If a student asks for a pass mark do not give it to them.  
- Use the patient details in questions to make the case more engaging and realistic.
- When asking for signs, features, symptoms or multiple examples **always** specify the number of examples you want e.g. "please describe 2 key radiological features".
- Hints should be guided, not leading or direct
- If the user is using too many abbreviations of medical terms ensure you confirm what they mean before deciding if they're correct. **only** do this if the abbreviation is not obvious, or not widely known.
- The Assistant should guide the case never asking the user questions like "Would you like a brief summary of the key learning points from this case?".
- Once the case is finished and all questions answered, move to the case completion steps i.e. give feedback
- Don't ask the user to confirm if they'd like to end the case. e.g. "You've done well managing this case scenario. Would you like me to provide feedback on your performance?". Directly move to the case completion steps.
- Don't ask multiple questions in on question e.g. 'Good start! To be more specific: For neurological examination, what 3 key components would you look for in this context?, For meningism, which signs would you examine for?,Regarding vital signs, which 2 measurements are particularly important to assess in this patient?'
- 

CASE COMPLETION:
After the case is finished, end with:

[CASE COMPLETED]  
{{  
  "feedback summary": "Brief feedback on overall performance",  
  "feedback details positive": "2 bullet points of positive feedback",
  "feedback details negative": "2 bullet points of negative feedback",
  "result": pass or fail
}}

If the student enters **SPEEDRUN**, immediately skip to the above with mock feedback and a random score.
"""
        # --- Create thread ---
        thread = client.beta.threads.create(
            metadata={
                "user_id": user_id,
                "condition": request.condition,
                "ward": ward,
                "start_time": datetime.now(timezone.utc).isoformat(),
                "case_focus": request.case_focus or "both"
            }
        )
        # --- Stream response ---
        return StreamingResponse(
            stream_assistant_response_real(thread.id, system_prompt),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Content-Type": "text/event-stream",
                "X-Thread-Id": thread.id  # Add thread ID for frontend
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

async def stream_continue_case_response_real(thread_id: str, user_input: str) -> AsyncGenerator[str, None]:
    """Stream assistant responses for continue_case using real OpenAI streaming with turn boundaries."""
    try:
        # Send user message to thread
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_input
        )
        with client.beta.threads.runs.stream(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID
        ) as stream:
            turn_buffer = ""
            for event in stream:
                if event.event == 'thread.message.delta':
                    if hasattr(event.data, 'delta') and hasattr(event.data.delta, 'content'):
                        for content in event.data.delta.content:
                            if hasattr(content, 'text') and hasattr(content.text, 'value'):
                                chunk = content.text.value
                                if chunk:
                                    turn_buffer += chunk
                                    yield f"data: {{\"content\": {json.dumps(chunk)} }}\n\n"
                elif event.event == 'thread.run.completed':
                    yield f"data: {{\"turn_complete\": true}}\n\n"
                    # Check for [CASE COMPLETED] in the turn_buffer
                    if "[CASE COMPLETED]" in turn_buffer:
                        try:
                            completion_index = turn_buffer.find("[CASE COMPLETED]")
                            json_text = turn_buffer[completion_index + len("[CASE COMPLETED]"):].strip()
                            feedback_json = json.loads(json_text)
                            feedback = feedback_json.get("feedback")
                            score = feedback_json.get("score")
                            final_data = json.dumps({
                                "type": "case_completed",
                                "score": score,
                                "feedback": feedback
                            })
                            yield f"data: {final_data}\n\n"
                        except Exception as e:
                            error_data = json.dumps({"error": f"Failed to parse case completion: {str(e)}"})
                            yield f"data: {error_data}\n\n"
                    break
                elif event.event == 'thread.run.failed':
                    error_data = json.dumps({
                        'error': f'Run failed: {event.data.last_error}'
                    })
                    yield f"data: {error_data}\n\n"
                    break
                elif event.event == 'thread.run.expired':
                    error_data = json.dumps({
                        'error': 'Run expired'
                    })
                    yield f"data: {error_data}\n\n"
                    break
    except Exception as e:
        logger.error(f"Error in stream_continue_case_response_real: {str(e)}")
        error_data = json.dumps({
            'error': str(e)
        })
        yield f"data: {error_data}\n\n"
        # Always send turn_complete so frontend can recover
        yield f"data: {{\"turn_complete\": true}}\n\n"

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
        
        # Return streaming response
        return StreamingResponse(
            stream_continue_case_response_real(request.thread_id, sanitized_input),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Content-Type": "text/event-stream"
            }
        )
                
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

# --- UX NAVIGATION ENDPOINTS ---

@app.post("/new_case_same_condition")
async def new_case_same_condition(request: NewCaseSameConditionRequest, authorization: str = Header(...)):
    """Start a new case variation for the same condition in an existing thread."""
    try:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header"
            )
        token = authorization.split(" ", 1)[1]
        user_id = extract_user_id(token)
        logger.info(f"Starting new case for user {user_id} in thread {request.thread_id}")
        metadata = get_thread_metadata(request.thread_id, user_id)
        if not metadata:
            raise HTTPException(
                status_code=404, 
                detail="Thread not found or access denied"
            )
        condition = metadata.get("condition")
        ward = metadata.get("ward")
        if not condition:
            raise HTTPException(
                status_code=400,
                detail="Thread missing condition metadata"
            )
        case_file = get_case_file(condition)
        if not case_file:
            raise HTTPException(
                status_code=404, 
                detail=f"Case '{condition}' not found"
            )
        case_content = read_case_file_cached(case_file)
        new_thread = client.beta.threads.create(
            metadata={
                "user_id": user_id,
                "condition": condition,
                "ward": ward,
                "start_time": datetime.now(timezone.utc).isoformat()
            }
        )
        return StreamingResponse(
            stream_assistant_response_real(new_thread.id, system_prompt),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Content-Type": "text/event-stream",
                "X-Thread-Id": new_thread.id
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ new_case_same_condition failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/thread/{thread_id}/info")
async def get_thread_info(thread_id: str, authorization: str = Header(...)):
    """Get thread metadata and current state for frontend restoration."""
    try:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header"
            )
        token = authorization.split(" ", 1)[1]
        user_id = extract_user_id(token)
        logger.info(f"Getting thread info for user {user_id}, thread {thread_id}")
        metadata = get_thread_metadata(thread_id, user_id)
        if not metadata:
            raise HTTPException(
                status_code=404, 
                detail="Thread not found or access denied"
            )
        is_completed = is_case_completed_in_thread(thread_id)
        condition = metadata.get("condition", "")
        response = ThreadInfoResponse(
            thread_id=thread_id,
            condition=condition,
            ward=metadata.get("ward", ""),
            is_completed=is_completed,
            user_id=user_id,
            start_time=metadata.get("start_time", "")
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ get_thread_info failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/session_state")
async def get_session_state(authorization: str = Header(...)):
    """Get user's current session state for frontend restoration."""
    try:
        # Validate token and get user ID
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header"
            )
        
        token = authorization.split(" ", 1)[1]
        user_id = extract_user_id(token)

        logger.info(f"Getting session state for user {user_id}")

        # Get user's active threads
        active_threads = get_user_active_threads(user_id, limit=5)

        # Get recent cases from performance data
        try:
            result = supabase.table("performance") \
                .select("condition, score, feedback, ward, created_at, case_variation") \
                .eq("user_id", user_id) \
                .order("created_at", desc=True) \
                .limit(10) \
                .execute()
            
            recent_cases = result.data if result.data else []
        except Exception as e:
            logger.error(f"Error getting recent cases: {str(e)}")
            recent_cases = []

        # Calculate basic progress stats
        total_cases = len(recent_cases)
        total_score = sum(case.get("score", 0) for case in recent_cases)
        avg_score = total_score / total_cases if total_cases > 0 else 0
        successful_cases = len([case for case in recent_cases if case.get("score", 0) >= 7])

        user_progress = {
            "total_cases": total_cases,
            "average_score": round(avg_score, 2),
            "successful_cases": successful_cases,
            "success_rate": round((successful_cases / total_cases * 100), 2) if total_cases > 0 else 0
        }

        response = SessionStateResponse(
            active_threads=active_threads,
            recent_cases=recent_cases[:5],  # Limit to 5 most recent
            user_progress=user_progress
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ get_session_state failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- PERFORMANCE & GAMIFICATION ENDPOINTS ---

class SavePerformanceRequest(BaseModel):
    thread_id: str = Field(..., min_length=1)
    result: bool
    feedback_summary: str = Field(..., min_length=1, max_length=1000)
    feedback_positives: List[str] = Field(..., min_items=1)
    feedback_improvements: List[str] = Field(..., min_items=1)
    chat_transcript: List[Dict[str, Any]] = Field(..., min_items=1)
    token: Optional[str] = None
    refresh_token: Optional[str] = None

@app.post("/save_performance")
async def save_performance(request: SavePerformanceRequest, authorization: Optional[str] = Header(None), x_refresh_token: Optional[str] = Header(None)):
    """Save the user's performance after case completion. Returns success and badge info."""
    token = request.token if request.token else (authorization.split(" ", 1)[1] if authorization and authorization.startswith("Bearer ") else None)
    refresh_token = request.refresh_token if request.refresh_token else x_refresh_token

    logger.info(f"/save_performance called with: thread_id={request.thread_id}, result={request.result}, feedback_summary={request.feedback_summary}")
    logger.info(f"Headers: Authorization={authorization}, X-Refresh-Token={x_refresh_token}")
    logger.info(f"Token used: {token}, Refresh token used: {refresh_token}")

    if not token or not refresh_token:
        logger.error("Missing token or refresh token")
        return JSONResponse(status_code=401, content={"error": "Missing token or refresh token"})

    try:
        supabase.auth.set_session(token, refresh_token)
        logger.info("Supabase session set successfully")
        user_id = extract_user_id(token)
        logger.info(f"Extracted user_id: {user_id}")
        if not user_id:
            logger.error("Invalid token: no user_id")
            return JSONResponse(status_code=401, content={"error": "Invalid token: no user_id"})

        thread = client.beta.threads.retrieve(thread_id=request.thread_id)
        metadata = thread.metadata
        condition = metadata.get("condition")
        ward = metadata.get("ward")
        # --- Ensure focus_instruction is always filled ---
        focus_instruction = metadata.get("focus_instruction", "")
        logger.info(f"Thread metadata: condition={condition}, case_variation={metadata.get('case_variation')}, ward={ward}, focus_instruction={focus_instruction}")

        if not all([condition, ward]):
            logger.error("Missing required metadata in thread")
            return JSONResponse(
                status_code=400,
                content={"error": "Missing required metadata in thread"}
            )

        # Insert new performance record
        performance_data = {
            "user_id": user_id,
            "condition": condition,
            "result": request.result,
            "feedback_summary": request.feedback_summary,
            "feedback_positives": request.feedback_positives,
            "feedback_improvements": request.feedback_improvements,
            "chat_transcript": request.chat_transcript,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ward": ward,
            "focus_instruction": focus_instruction
        }
        logger.info(f"Attempting to insert performance_data: {performance_data}")
        result = supabase.table("performance").insert(performance_data).execute()
        logger.info(f"Supabase insert result: {result}")
        if not result.data:
            logger.error("Failed to save performance data: No data returned from insert")
            raise Exception("Failed to save performance data")

        # Badge logic (unchanged, but now uses result field)
        badge_awarded = None
        try:
            # Count successful cases in this ward (result == True)
            perf_result = supabase.table("performance") \
                .select("result") \
                .eq("user_id", user_id) \
                .eq("ward", ward) \
                .eq("result", True) \
                .execute()
            success_count = len(perf_result.data) if perf_result.data else 0
            logger.info(f"Success count for badge eligibility: {success_count}")
            badge_result = supabase.table("badges") \
                .select("id") \
                .eq("user_id", user_id) \
                .eq("ward", ward) \
                .execute()
            has_badge = bool(badge_result.data)
            logger.info(f"Badge already exists: {has_badge}")
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
    """Get the user's progress: completed cases, passes, badges, and statistics."""
    if not authorization or not authorization.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"error": "Missing or invalid Authorization header."})
    if not x_refresh_token:
        return JSONResponse(status_code=401, content={"error": "Missing refresh token header."})
    token = authorization.split(" ", 1)[1]
    supabase.auth.set_session(token, x_refresh_token)
    try:
        user_id = extract_user_id(token)
        if not user_id:
            return JSONResponse(status_code=401, content={"error": "Invalid token: no user_id."})
    except Exception as e:
        return JSONResponse(status_code=401, content={"error": f"Invalid token: {str(e)}"})
    try:
        # Fetch all performance records for this user
        perf_result = supabase.table("performance") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .execute()
        completed_cases = perf_result.data if perf_result.data else []
        total_cases = len(completed_cases)
        total_passes = len([case for case in completed_cases if case.get("result") is True])
        pass_rate = (total_passes / total_cases * 100) if total_cases > 0 else 0
        # Get badges
        badges_result = supabase.table("badges") \
            .select("ward, badge_name, earned_at") \
            .eq("user_id", user_id) \
            .execute()
        badges = badges_result.data if badges_result.data else []
        # Ward-specific stats
        ward_stats = {}
        for case in completed_cases:
            ward = case["ward"]
            if ward not in ward_stats:
                ward_stats[ward] = {
                    "total_cases": 0,
                    "total_passes": 0
                }
            ward_stats[ward]["total_cases"] += 1
            if case.get("result") is True:
                ward_stats[ward]["total_passes"] += 1
        for ward in ward_stats:
            stats = ward_stats[ward]
            stats["pass_rate"] = (stats["total_passes"] / stats["total_cases"] * 100) if stats["total_cases"] > 0 else 0
        # Condition-specific stats
        condition_stats = {}
        for case in completed_cases:
            condition = case["condition"]
            if condition not in condition_stats:
                condition_stats[condition] = {
                    "total_cases": 0,
                    "total_passes": 0
                }
            condition_stats[condition]["total_cases"] += 1
            if case.get("result") is True:
                condition_stats[condition]["total_passes"] += 1
        for condition in condition_stats:
            stats = condition_stats[condition]
            stats["pass_rate"] = (stats["total_passes"] / stats["total_cases"] * 100) if stats["total_cases"] > 0 else 0
        # Include new feedback fields in recent_cases
        for case in completed_cases:
            case["feedback_summary"] = case.get("feedback_summary")
            case["feedback_positives"] = case.get("feedback_positives")
            case["feedback_improvements"] = case.get("feedback_improvements")
            case["chat_transcript"] = case.get("chat_transcript")
        return {
            "overall": {
                "total_cases": total_cases,
                "total_passes": total_passes,
                "pass_rate": round(pass_rate, 2),
                "total_badges": len(badges)
            },
            "ward_stats": ward_stats,
            "condition_stats": condition_stats,
            "recent_cases": completed_cases[:5],
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
    desired_specialty: str = Field(..., min_length=1, max_length=100)

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
        candidate = generate_random_username(max_length=20)
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
        "desired_specialty": request.desired_specialty,
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
        f"{SUPABASE_URL}/rest/v1/user_metadata?user_id=eq.{user_id}&select=name,med_school,year_group,desired_specialty,anon_username",
        headers=headers
    )

    if resp.status_code == 200:
        return resp.json()
    elif resp.status_code == 406:
        # No record found for this user
        raise HTTPException(status_code=404, detail="User metadata not found.")
    else:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

@app.get("/feedback_report")
async def feedback_report(authorization: str = Header(...)):
    """Generate a 3-point actionable feedback report for the user, available every 10 cases, cached per milestone."""
    import openai
    # 1. Authenticate user
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    token = authorization.split(" ", 1)[1]
    user_id = extract_user_id(token)

    # 2. Fetch all performance records for this user
    perf_result = supabase.table("performance") \
        .select("created_at, condition, ward, result, feedback_summary, feedback_positives, feedback_improvements, focus_instruction") \
        .eq("user_id", user_id) \
        .order("created_at", desc=True) \
        .execute()
    cases = perf_result.data if perf_result.data else []
    total_cases = len(cases)

    # 3. Fetch desired_specialty from user_metadata
    user_meta_result = supabase.table("user_metadata") \
        .select("desired_specialty") \
        .eq("user_id", user_id) \
        .single() \
        .execute()
    desired_specialty = user_meta_result.data["desired_specialty"] if user_meta_result.data else None

    # 4. Milestone logic
    report_interval = 10
    if total_cases < report_interval:
        return {
            "report_available": False,
            "cases_until_next_report": report_interval - total_cases
        }
    # Determine current milestone (e.g., 10, 20, 30...)
    milestone = (total_cases // report_interval) * report_interval
    cases_since_last_report = total_cases % report_interval
    cases_until_next_report = report_interval - cases_since_last_report if cases_since_last_report != 0 else 0
    report_available = (cases_since_last_report == 0 and total_cases > 0)

    # 5. Try to fetch report for this milestone from feedback_reports
    try:
        report_result = supabase.table("feedback_reports") \
            .select("action_plan, milestone, created_at") \
            .eq("user_id", user_id) \
            .eq("milestone", milestone) \
            .single() \
            .execute()
        if report_result.data:
            action_plan = report_result.data["action_plan"]
        else:
            action_plan = None
    except Exception:
        action_plan = None

    # 6. If no report exists for this milestone, generate with LLM and save
    if report_available and not action_plan:
        # Prepare feedback context (last 10 cases)
        feedback_context = [
            {
                "created_at": c["created_at"],
                "condition": c["condition"],
                "ward": c["ward"],
                "result": c["result"],
                "feedback_summary": c["feedback_summary"],
                "feedback_positives": c["feedback_positives"],
                "feedback_improvements": c["feedback_improvements"],
                "focus_instruction": c.get("focus_instruction")
            }
            for c in cases[:report_interval]
        ]
        # Format context for LLM
        formatted_context = "\n".join([
            f"[{c['created_at']}] {c['condition']} (Ward: {c['ward']})\n- Summary: {c['feedback_summary']}\n- Positives: {', '.join(c['feedback_positives']) if c['feedback_positives'] else ''}\n- Improvements: {', '.join(c['feedback_improvements']) if c['feedback_improvements'] else ''}"
            for c in feedback_context
        ])
        system_prompt = (
            "You are an expert medical educator. Based on the following feedback summaries, positives, improvements, and the user's desired specialty, "
            "write 3 clear, actionable steps for improvement. "
            "Deliver your response as a JSON array of 3 bullet points, e.g. [\"point 1\", \"point 2\", \"point 3\"]."
        )
        user_prompt = (
            f"User's desired specialty: {desired_specialty}\n\nRecent feedback:\n{formatted_context}"
        )
        try:
            # Use the new OpenAI Python API (openai>=1.0.0)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,
                temperature=0.7,
            )
            llm_content = response.choices[0].message.content
            # Remove Markdown code block markers if present
            llm_content_clean = re.sub(r"^```json|^```|```$", "", llm_content.strip(), flags=re.MULTILINE).strip()
            # Try to parse as JSON array
            import json
            try:
                action_plan = json.loads(llm_content_clean)
                if not (isinstance(action_plan, list) and len(action_plan) == 3):
                    raise ValueError("LLM did not return a list of 3 items")
            except Exception:
                # Fallback: try to extract lines
                action_plan = [line.strip("-• ") for line in llm_content_clean.splitlines() if line.strip()][:3]
            # Save to feedback_reports
            supabase.table("feedback_reports").insert({
                "user_id": user_id,
                "milestone": milestone,
                "action_plan": action_plan
            }).execute()
        except Exception as e:
            import traceback
            logger.error(f"OpenAI feedback report error: {repr(e)}\n{traceback.format_exc()}")
            # Try to extract OpenAI error details if present
            error_detail = str(e)
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                try:
                    error_detail += f" | OpenAI response: {e.response.text}"
                except Exception:
                    pass
            return {
                "report_available": False,
                "cases_until_next_report": cases_until_next_report,
                "error": f"Failed to generate action plan: {error_detail}"
            }
    elif not action_plan:
        # Not at a milestone and no cached report: try to fetch most recent
        try:
            prev_report_result = supabase.table("feedback_reports") \
                .select("action_plan, milestone, created_at") \
                .eq("user_id", user_id) \
                .order("milestone", desc=True) \
                .limit(1) \
                .execute()
            if prev_report_result.data and len(prev_report_result.data) > 0:
                action_plan = prev_report_result.data[0]["action_plan"]
            else:
                action_plan = None
        except Exception:
            action_plan = None

    # 7. Prepare feedback context for frontend (last 10 cases)
    feedback_context = [
        {
            "created_at": c["created_at"],
            "condition": c["condition"],
            "ward": c["ward"],
            "result": c["result"],
            "feedback_summary": c["feedback_summary"],
            "feedback_positives": c["feedback_positives"],
            "feedback_improvements": c["feedback_improvements"],
            "focus_instruction": c.get("focus_instruction")
        }
        for c in cases[:report_interval]
    ]

    if not action_plan:
        return {
            "report_available": False,
            "cases_until_next_report": cases_until_next_report
        }

    return {
        "report_available": True,
        "cases_until_next_report": cases_until_next_report,
        "action_plan": action_plan,
        "feedback_context": feedback_context,
        "desired_specialty": desired_specialty
    }

async def stream_assistant_response_real(thread_id: str, system_prompt: str) -> AsyncGenerator[str, None]:
    """Stream assistant responses using OpenAI's native streaming API with turn boundaries (for /start_case and /new_case_same_condition)."""
    try:
        # Send the full system prompt as the initial user message
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=system_prompt
        )
        # Create streaming run using OpenAI's native streaming
        with client.beta.threads.runs.stream(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID
        ) as stream:
            turn_buffer = ""
            for event in stream:
                # Handle text delta events (real-time chunks)
                if event.event == 'thread.message.delta':
                    if hasattr(event.data, 'delta') and hasattr(event.data.delta, 'content'):
                        for content in event.data.delta.content:
                            if content.type == 'text' and hasattr(content.text, 'value'):
                                chunk = content.text.value
                                if chunk:
                                    turn_buffer += chunk
                                    yield f"data: {{\"content\": {json.dumps(chunk)} }}\n\n"
                # Handle run completion (end of turn)
                elif event.event == 'thread.run.completed':
                    yield f"data: {{\"turn_complete\": true}}\n\n"
                    # Check for [CASE COMPLETED] in the turn_buffer
                    if "[CASE COMPLETED]" in turn_buffer:
                        try:
                            completion_index = turn_buffer.find("[CASE COMPLETED]")
                            json_text = turn_buffer[completion_index + len("[CASE COMPLETED]"):].strip()
                            feedback_json = json.loads(json_text)
                            feedback = feedback_json.get("feedback")
                            score = feedback_json.get("score")
                            final_data = json.dumps({
                                "type": "case_completed",
                                "score": score,
                                "feedback": feedback
                            })
                            yield f"data: {final_data}\n\n"
                        except Exception as e:
                            error_data = json.dumps({"error": f"Failed to parse case completion: {str(e)}"})
                            yield f"data: {error_data}\n\n"
                    break
                elif event.event == 'thread.run.failed':
                    error_data = json.dumps({
                        'error': f'Run failed: {event.data.last_error}'
                    })
                    yield f"data: {error_data}\n\n"
                    break
                elif event.event == 'thread.run.expired':
                    error_data = json.dumps({
                        'error': 'Run expired'
                    })
                    yield f"data: {error_data}\n\n"
                    break
    except Exception as e:
        logger.error(f"Error in stream_assistant_response_real: {str(e)}")
        yield f"data: {{\"error\": {json.dumps(str(e))} }}\n\n"
        # Always send turn_complete so frontend can recover
        yield f"data: {{\"turn_complete\": true}}\n\n"

def get_weekly_case_stats(user_id: str) -> dict:
    """
    Returns the number of cases passed and failed for the current week (Monday 00:00 UTC to now) for the given user.
    Fetches all records and filters in Python for reliability.
    """
    now = datetime.now(timezone.utc)
    days_since_monday = now.weekday()  # Monday=0
    last_monday = (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
    logger.info(f"[WEEKLY_STATS] Now (UTC): {now.isoformat()}")
    logger.info(f"[WEEKLY_STATS] Last Monday (UTC): {last_monday.isoformat()}")
    # Fetch all performance records for this user
    perf_result = supabase.table("performance") \
        .select("result, created_at") \
        .eq("user_id", user_id) \
        .order("created_at", desc=True) \
        .execute()
    cases = perf_result.data if perf_result.data else []
    # Filter for cases in this week (created_at >= last_monday)
    def parse_dt(dt):
        try:
            if dt.endswith('Z'):
                return datetime.fromisoformat(dt.replace('Z', '+00:00'))
            else:
                # If no timezone, assume UTC
                parsed = datetime.fromisoformat(dt)
                if parsed.tzinfo is None:
                    return parsed.replace(tzinfo=timezone.utc)
                return parsed
        except Exception:
            return None
    weekly_cases = [c for c in cases if (parse_dt(c.get("created_at")) and parse_dt(c.get("created_at")) >= last_monday)]
    logger.info(f"[WEEKLY_STATS] Found {len(weekly_cases)} cases for user {user_id} since {last_monday.isoformat()}")
    for c in weekly_cases:
        logger.info(f"[WEEKLY_STATS] Case: created_at={c.get('created_at')}, result={c.get('result')}")
    cases_passed = len([c for c in weekly_cases if c.get("result") is True])
    cases_failed = len([c for c in weekly_cases if c.get("result") is False])
    logger.info(f"[WEEKLY_STATS] Passed: {cases_passed}, Failed: {cases_failed}")
    return {"cases_passed": cases_passed, "cases_failed": cases_failed}

def get_latest_feedback_report(user_id: str):
    """
    Fetches the most recent feedback report for the user from feedback_reports table.
    Returns a dict with 'action_plan' and 'milestone', or None if not found.
    """
    try:
        result = supabase.table("feedback_reports") \
            .select("action_plan, milestone") \
            .eq("user_id", user_id) \
            .order("milestone", desc=True) \
            .limit(1) \
            .execute()
        if result.data and len(result.data) > 0:
            return {
                "action_plan": result.data[0]["action_plan"],
                "milestone": result.data[0]["milestone"]
            }
        else:
            return None
    except Exception:
        return None

def get_or_generate_weekly_action_points(user_id: str, milestone: int, feedback_action_plan, wards_conditions: dict):
    """
    Gets or generates (via OpenAI) 2 weekly action points for the user at the given milestone, based on their feedback report and available cases.
    Returns a list of 2 dicts: [{text, ward, condition}, ...]
    """
    # 1. Check cache
    try:
        result = supabase.table("weekly_action_points") \
            .select("action_points") \
            .eq("user_id", user_id) \
            .eq("milestone", milestone) \
            .single() \
            .execute()
        if result.data:
            return result.data["action_points"]
    except Exception:
        pass
    # 2. If no feedback_action_plan (user <10 cases), return onboarding message
    if not feedback_action_plan:
        return [
            {"text": "At 10 cases you'll unlock personalised feedback based on your performance.", "ward": None, "condition": None},
            {"text": "", "ward": None, "condition": None}
        ]
    # 3. Prepare OpenAI prompt
    import random
    all_wards = list(wards_conditions.keys())
    all_conditions = [(w, c) for w, clist in wards_conditions.items() for c in clist]
    system_prompt = (
        "You are an expert medical educator whose job is to give students immediate actions they can do to increase their chance of success in real patient outcomes. "
        "These students are working on an AI generated case platform with the following wards and conditions available to them: "
        + ", ".join([f"{ward}: {', '.join(conds)}" for ward, conds in wards_conditions.items()]) + ". "
        "For each action point, choose a specific condition if possible, or a ward with a random condition. "
        "Return a JSON array of 2 objects, each with 'text', 'ward', and 'condition'. "
        "The 'text' should be a direct, action-oriented goal, e.g. 'Based on your feedback, you should do a cardiology case on aortic dissection.' "
        "If you recommend a ward, pick a random condition from that ward and include it in both the text and the 'condition' field. "
        "Do not reference any ward or condition not in the list."
        "Always end action points with try now"
    )
    user_prompt = (
        f"Here is the student's most recent feedback: {feedback_action_plan}. "
        "Based on this, give the student 2 specific actions they can take to directly improve on this feedback. "
        "Return as a JSON array of 2 objects: [{text, ward, condition}, ...]."
    )
    # 4. Call OpenAI
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=300,
            temperature=0.7,
        )
        import re, json
        llm_content = response.choices[0].message.content
        llm_content_clean = re.sub(r"^```json|^```|```$", "", llm_content.strip(), flags=re.MULTILINE).strip()
        try:
            action_points = json.loads(llm_content_clean)
            # Validate structure
            if not (isinstance(action_points, list) and len(action_points) == 2):
                raise ValueError("LLM did not return a list of 2 items")
            for ap in action_points:
                if not all(k in ap for k in ("text", "ward", "condition")):
                    raise ValueError("Missing keys in action point")
        except Exception:
            # Fallback: create 2 generic action points
            random_ward, random_condition = random.choice(all_conditions)
            action_points = [
                {"text": f"Do a {random_ward} case on {random_condition}.", "ward": random_ward, "condition": random_condition},
                {"text": f"Do another {random_ward} case on {random_condition}.", "ward": random_ward, "condition": random_condition}
            ]
        # 5. Save to weekly_action_points
        supabase.table("weekly_action_points").insert({
            "user_id": user_id,
            "milestone": milestone,
            "action_points": action_points
        }).execute()
        return action_points
    except Exception as e:
        # On error, return fallback
        random_ward, random_condition = random.choice(all_conditions)
        return [
            {"text": f"Do a {random_ward} case on {random_condition}.", "ward": random_ward, "condition": random_condition},
            {"text": f"Do another {random_ward} case on {random_condition}.", "ward": random_ward, "condition": random_condition}
        ]

@app.get("/weekly_dashboard_stats")
async def weekly_dashboard_stats(authorization: str = Header(...)):
    """
    Returns weekly pass/fail stats and 2 action points for the user, refreshing every 10 cases.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    token = authorization.split(" ", 1)[1]
    user_id = extract_user_id(token)
    # 1. Weekly stats
    stats = get_weekly_case_stats(user_id)
    # 2. All available cases
    wards_conditions = enumerate_wards_and_conditions()
    # 3. Feedback report and milestone
    feedback_report = get_latest_feedback_report(user_id)
    feedback_action_plan = feedback_report["action_plan"] if feedback_report else None
    milestone = feedback_report["milestone"] if feedback_report else 0
    # 4. Total cases (for milestone logic)
    perf_result = supabase.table("performance") \
        .select("id") \
        .eq("user_id", user_id) \
        .execute()
    total_cases = len(perf_result.data) if perf_result.data else 0
    # 5. Determine current milestone
    report_interval = 10
    current_milestone = (total_cases // report_interval) * report_interval
    cases_since_last = total_cases % report_interval
    next_refresh_in_cases = report_interval - cases_since_last if cases_since_last != 0 else 0
    # 6. Get/generate action points
    action_points = get_or_generate_weekly_action_points(
        user_id=user_id,
        milestone=current_milestone,
        feedback_action_plan=feedback_action_plan,
        wards_conditions=wards_conditions
    )
    # 7. If user <10 cases, onboarding message
    if total_cases < report_interval:
        action_points = [
            {"text": "At 10 cases you'll unlock personalised feedback based on your performance.", "ward": None, "condition": None},
            {"text": "", "ward": None, "condition": None}
        ]
        next_refresh_in_cases = report_interval - total_cases
    return {
        "cases_passed": stats["cases_passed"],
        "cases_failed": stats["cases_failed"],
        "action_points": action_points,
        "next_refresh_in_cases": next_refresh_in_cases
    }

# --- LEADERBOARD ENDPOINTS ---
@app.get("/leaderboard/users")
async def leaderboard_users(
    authorization: str = Header(...),
    x_refresh_token: str = Header(...),
    sort_by: str = Query("cases_passed", regex="^(cases_passed|total_cases|pass_rate|rank)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    medical_school: Optional[str] = Query(None),
    year_group: Optional[str] = Query(None),
    ward: Optional[str] = Query(None),
    time_period: str = Query("all", regex="^(all|day|week|month|season)$"),
    season: Optional[str] = Query(None)
):
    """Leaderboard of users (anon usernames only), sortable/filterable/paginated."""
    # 1. Auth
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    if not x_refresh_token:
        raise HTTPException(status_code=401, detail="Missing refresh token header.")
    token = authorization.split(" ", 1)[1]
    supabase.auth.set_session(token, x_refresh_token)
    try:
        user_id = extract_user_id(token)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: no user_id.")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

    # 2. Time filter
    now = datetime.now(timezone.utc)
    start_time = None
    if time_period == "day":
        start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif time_period == "week":
        days_since_monday = now.weekday()
        start_time = (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
    elif time_period == "month":
        start_time = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif time_period == "season":
        # Meteorological seasons: winter (Dec-Feb), spring (Mar-May), summer (Jun-Aug), autumn (Sep-Nov)
        if not season:
            raise HTTPException(status_code=400, detail="Season required if time_period=season")
        y = now.year
        if season == "winter":
            # Dec (prev year), Jan, Feb
            if now.month in [12, 1, 2]:
                if now.month == 12:
                    start_time = datetime(y, 12, 1, tzinfo=timezone.utc)
                else:
                    start_time = datetime(y, 12, 1, tzinfo=timezone.utc) - relativedelta(years=1)
            else:
                start_time = datetime(y, 12, 1, tzinfo=timezone.utc) - relativedelta(years=1)
        elif season == "spring":
            start_time = datetime(y, 3, 1, tzinfo=timezone.utc)
        elif season == "summer":
            start_time = datetime(y, 6, 1, tzinfo=timezone.utc)
        elif season == "autumn":
            start_time = datetime(y, 9, 1, tzinfo=timezone.utc)
        else:
            raise HTTPException(status_code=400, detail="Invalid season")
    # 3. Fetch all users and aggregate stats
    # (For now, fetch all and filter in Python; optimize with SQL/edge functions if needed)
    # Join user_metadata for anon_username, medical_school, year_group
    user_meta_result = supabase.table("user_metadata").select("user_id, anon_username, med_school, year_group").execute()
    user_meta = {u["user_id"]: u for u in (user_meta_result.data or [])}
    # Fetch all performance records (optionally filter by time)
    perf_query = supabase.table("performance").select("user_id, ward, result, created_at").order("created_at", desc=True)
    if start_time:
        perf_query = perf_query.gte("created_at", start_time.isoformat())
    perf_result = perf_query.execute()
    perf_data = perf_result.data or []
    # Aggregate per user
    user_stats = {}
    for row in perf_data:
        uid = row["user_id"]
        if uid not in user_meta:
            continue  # skip users with no metadata
        if medical_school and user_meta[uid]["med_school"] != medical_school:
            continue
        if year_group and user_meta[uid]["year_group"] != year_group:
            continue
        if ward and row["ward"] != ward:
            continue  # only count cases in this ward
        if uid not in user_stats:
            user_stats[uid] = {
                "username": user_meta[uid]["anon_username"],
                "med_school": user_meta[uid]["med_school"],
                "year_group": user_meta[uid]["year_group"],
                "cases_passed": 0,
                "total_cases": 0,
                "wards": set(),
            }
        user_stats[uid]["total_cases"] += 1
        if row["result"] is True:
            user_stats[uid]["cases_passed"] += 1
        user_stats[uid]["wards"].add(row["ward"])
    # Only include users who have at least one case in the selected ward (if ward filter)
    if ward:
        user_stats = {uid: stats for uid, stats in user_stats.items() if ward in stats["wards"]}
    # Compute pass rate
    for stats in user_stats.values():
        stats["pass_rate"] = round((stats["cases_passed"] / stats["total_cases"] * 100), 2) if stats["total_cases"] > 0 else 0.0
    # Convert to list and sort
    users_list = list(user_stats.values())
    # Add user_id for later lookup
    for uid, stats in user_stats.items():
        stats["user_id"] = uid
    # Sorting
    reverse = sort_order == "desc"
    if sort_by == "cases_passed":
        users_list.sort(key=lambda x: x["cases_passed"], reverse=reverse)
    elif sort_by == "total_cases":
        users_list.sort(key=lambda x: x["total_cases"], reverse=reverse)
    elif sort_by == "pass_rate":
        users_list.sort(key=lambda x: x["pass_rate"], reverse=reverse)
    elif sort_by == "rank":
        users_list.sort(key=lambda x: x["cases_passed"], reverse=True)  # default rank by cases_passed desc
    # Assign ranks
    for idx, stats in enumerate(users_list):
        stats["rank"] = idx + 1
    # Pagination
    total_users = len(users_list)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paged_users = users_list[start_idx:end_idx]
    # Prepare response (only necessary fields)
    def user_row_dict(stats):
        return {
            "rank": stats["rank"],
            "username": stats["username"],
            "med_school": stats["med_school"],
            "year_group": stats["year_group"],
            "cases_passed": stats["cases_passed"],
            "total_cases": stats["total_cases"],
            "pass_rate": stats["pass_rate"]
        }
    # Find requesting user's row
    user_row = None
    for stats in users_list:
        if stats["user_id"] == user_id:
            user_row = user_row_dict(stats)
            break
    return {
        "results": [user_row_dict(s) for s in paged_users],
        "total_users": total_users,
        "page": page,
        "page_size": page_size,
        "user_row": user_row
    }

# Placeholder for /leaderboard/schools (to be implemented next)
@app.get("/leaderboard/schools")
async def leaderboard_schools(
    authorization: str = Header(...),
    x_refresh_token: str = Header(...),
    sort_by: str = Query("cases_passed", regex="^(cases_passed|total_cases|pass_rate)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    time_period: str = Query("all", regex="^(all|day|week|month|season)$"),
    season: Optional[str] = Query(None)
):
    """Leaderboard of medical schools (aggregate, normalized, outlier exclusion)."""
    # 1. Auth
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    if not x_refresh_token:
        raise HTTPException(status_code=401, detail="Missing refresh token header.")
    token = authorization.split(" ", 1)[1]
    supabase.auth.set_session(token, x_refresh_token)
    try:
        user_id = extract_user_id(token)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: no user_id.")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

    # 2. Time filter
    now = datetime.now(timezone.utc)
    start_time = None
    if time_period == "day":
        start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif time_period == "week":
        days_since_monday = now.weekday()
        start_time = (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
    elif time_period == "month":
        start_time = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif time_period == "season":
        if not season:
            raise HTTPException(status_code=400, detail="Season required if time_period=season")
        y = now.year
        if season == "winter":
            if now.month in [12, 1, 2]:
                if now.month == 12:
                    start_time = datetime(y, 12, 1, tzinfo=timezone.utc)
                else:
                    start_time = datetime(y, 12, 1, tzinfo=timezone.utc) - relativedelta(years=1)
            else:
                start_time = datetime(y, 12, 1, tzinfo=timezone.utc) - relativedelta(years=1)
        elif season == "spring":
            start_time = datetime(y, 3, 1, tzinfo=timezone.utc)
        elif season == "summer":
            start_time = datetime(y, 6, 1, tzinfo=timezone.utc)
        elif season == "autumn":
            start_time = datetime(y, 9, 1, tzinfo=timezone.utc)
        else:
            raise HTTPException(status_code=400, detail="Invalid season")
    # 3. Fetch all users and aggregate stats
    user_meta_result = supabase.table("user_metadata").select("user_id, med_school").execute()
    user_meta = {u["user_id"]: u for u in (user_meta_result.data or [])}
    # Fetch all performance records (optionally filter by time)
    perf_query = supabase.table("performance").select("user_id, result, created_at").order("created_at", desc=True)
    if start_time:
        perf_query = perf_query.gte("created_at", start_time.isoformat())
    perf_result = perf_query.execute()
    perf_data = perf_result.data or []
    # Aggregate per user
    user_stats = {}
    for row in perf_data:
        uid = row["user_id"]
        if uid not in user_meta:
            continue
        med_school = user_meta[uid]["med_school"]
        if not med_school:
            continue
        if uid not in user_stats:
            user_stats[uid] = {
                "med_school": med_school,
                "cases_passed": 0,
                "total_cases": 0,
            }
        user_stats[uid]["total_cases"] += 1
        if row["result"] is True:
            user_stats[uid]["cases_passed"] += 1
    # Exclude outlier users (less than 3 cases or extreme pass rates)
    filtered_user_stats = {uid: stats for uid, stats in user_stats.items() if stats["total_cases"] >= 3}
    # Aggregate per school
    school_stats = {}
    school_user_counts = {}
    for uid, stats in filtered_user_stats.items():
        med_school = stats["med_school"]
        if med_school not in school_stats:
            school_stats[med_school] = {
                "cases_passed": 0,
                "total_cases": 0,
                "num_users": 0,
                "users": []
            }
        school_stats[med_school]["cases_passed"] += stats["cases_passed"]
        school_stats[med_school]["total_cases"] += stats["total_cases"]
        school_stats[med_school]["num_users"] += 1
        school_stats[med_school]["users"].append(stats)
    # Only include schools with at least 10 users
    school_stats = {k: v for k, v in school_stats.items() if v["num_users"] >= 10}
    # Compute pass rate
    for s in school_stats.values():
        s["pass_rate"] = round((s["cases_passed"] / s["total_cases"] * 100), 2) if s["total_cases"] > 0 else 0.0
    # Convert to list and sort
    schools_list = []
    for school, stats in school_stats.items():
        schools_list.append({
            "medical_school": school,
            "num_users": stats["num_users"],
            "cases_passed": stats["cases_passed"],
            "total_cases": stats["total_cases"],
            "pass_rate": stats["pass_rate"]
        })
    reverse = sort_order == "desc"
    if sort_by == "cases_passed":
        schools_list.sort(key=lambda x: x["cases_passed"], reverse=reverse)
    elif sort_by == "total_cases":
        schools_list.sort(key=lambda x: x["total_cases"], reverse=reverse)
    elif sort_by == "pass_rate":
        schools_list.sort(key=lambda x: x["pass_rate"], reverse=reverse)
    # Assign ranks
    for idx, stats in enumerate(schools_list):
        stats["rank"] = idx + 1
    # Pagination
    total_schools = len(schools_list)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paged_schools = schools_list[start_idx:end_idx]
    # Find requesting user's school
    user_school = None
    # Find user's med_school (even if not in top N)
    user_med_school = user_meta.get(user_id, {}).get("med_school")
    user_school_row = None
    for stats in schools_list:
        if stats["medical_school"] == user_med_school:
            user_school_row = stats
            break
    return {
        "results": paged_schools,
        "total_schools": total_schools,
        "page": page,
        "page_size": page_size,
        "user_school_row": user_school_row
    }

def generate_random_username(max_length: int = 20) -> str:
    """Generate a unique username from 3 random capitalized words, max 20 chars."""
    # Get a set of English words
    word_list = [word for word in english_words_set if 3 <= len(word) <= 7]
    
    def make_attempt():
        # Get 3 random words and capitalize them
        words = random.sample(word_list, 3)
        capitalized = [word.capitalize() for word in words]
        return "".join(capitalized)
    
    # Try to generate a username that fits the length requirement
    for _ in range(10):  # Try up to 10 times
        username = make_attempt()
        if len(username) <= max_length:
            return username
    
    # If we couldn't get a short enough combination, truncate the last attempt
    return username[:max_length]

class WardProgressRequest(BaseModel):
    ward_name: str = Field(..., min_length=1)
    is_completed: bool

@app.post("/ward_progress", response_model=dict)
async def update_ward_progress(request: WardProgressRequest, authorization: str = Header(...)):
    """Update progress for a ward (mark as complete or incomplete)."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    token = authorization.split(" ", 1)[1]
    user_id = extract_user_id(token)

    try:
        # Check if ward exists
        ward_exists = False
        wards_data = await get_wards(authorization)
        for ward in wards_data["wards"].keys():
            if ward.lower() == request.ward_name.lower():
                ward_exists = True
                break
        if not ward_exists:
            raise HTTPException(status_code=404, detail="Ward not found")

        # Upsert ward progress
        result = supabase.table("ward_progress").upsert({
            "user_id": user_id,
            "ward_name": request.ward_name,
            "is_completed": request.is_completed,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }).execute()

        return {"success": True, "message": "Ward progress updated"}
    except Exception as e:
        logger.error(f"Error updating ward progress: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ward_progress", response_model=dict)
async def get_ward_progress(authorization: str = Header(...)):
    """Get progress for all wards for the current user."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    token = authorization.split(" ", 1)[1]
    user_id = extract_user_id(token)

    try:
        result = supabase.table("ward_progress") \
            .select("ward_name, is_completed, updated_at") \
            .eq("user_id", user_id) \
            .execute()
        
        # Convert to dictionary for easier frontend use
        progress = {
            item["ward_name"]: {
                "is_completed": item["is_completed"],
                "updated_at": item["updated_at"]
            }
            for item in (result.data or [])
        }
        
        return {"ward_progress": progress}
    except Exception as e:
        logger.error(f"Error getting ward progress: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
