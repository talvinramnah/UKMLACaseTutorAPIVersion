import streamlit as st
import requests
import json
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONFIG ---
# Load environment variables from .env file if it exists
load_dotenv()

# Get backend URL from environment variable or use default
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="UKMLA Case Tutor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE INITIALIZATION ---
def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "user": None,
        "access_token": None,
        "refresh_token": None,
        "token_expiry": None,
        "thread_id": None,
        "case_started": False,
        "chat_history": [],
        "condition": None,
        "error_message": None,
        "is_loading": False,
        "case_completed": False,
        "show_goodbye": False,
        "score": None,
        "feedback": None,
        "case_variation": None,
        "total_score": 0
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# --- AUTHENTICATION HELPERS ---
def signup_user(email: str, password: str) -> dict:
    """Register a new user account."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/signup",
            json={"email": email, "password": password}
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def login_user(email: str, password: str) -> dict:
    """Authenticate user and get tokens."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/login",
            json={"email": email, "password": password}
        )
        if response.status_code == 200:
            data = response.json()
            # Store tokens and user info in session state
            st.session_state.access_token = data["access_token"]
            st.session_state.refresh_token = data["refresh_token"]
            st.session_state.user = data["user"]
            st.session_state.token_expiry = datetime.now() + timedelta(hours=1)
            return {"success": True}
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def refresh_token() -> bool:
    """Refresh the access token using the refresh token."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/refresh",
            json={"refresh_token": st.session_state.refresh_token}
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.access_token = data["access_token"]
            st.session_state.refresh_token = data["refresh_token"]
            st.session_state.token_expiry = datetime.now() + timedelta(hours=1)
            return True
        return False
    except Exception:
        return False

def logout_user():
    """Logout user and clear session state."""
    try:
        if st.session_state.access_token:
            requests.post(
                f"{BACKEND_URL}/logout",
                json={"token": st.session_state.access_token}
            )
    except Exception:
        pass  # Proceed with local logout even if server logout fails
    
    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()

def check_token_expiry():
    """Check if the access token is expired and refresh if needed."""
    if not st.session_state.token_expiry:
        return False
    
    if datetime.now() >= st.session_state.token_expiry:
        return refresh_token()
    return True

# --- API HELPERS ---
def make_authenticated_request(method, endpoint, **kwargs):
    """Make an authenticated request to the backend API."""
    if not st.session_state.get("access_token") or not st.session_state.get("refresh_token"):
        st.error("No authentication tokens found. Please log in again.")
        logout_user()
        st.rerun()

    headers = {
        "Authorization": f"Bearer {st.session_state.access_token}",
        "X-Refresh-Token": st.session_state.refresh_token
    }
    
    if "headers" in kwargs:
        kwargs["headers"].update(headers)
    else:
        kwargs["headers"] = headers

    try:
        response = getattr(requests, method)(f"{BACKEND_URL}{endpoint}", **kwargs)
        
        # Check for new tokens in response headers
        new_access_token = response.headers.get("X-New-Access-Token")
        new_refresh_token = response.headers.get("X-New-Refresh-Token")
        
        if new_access_token and new_refresh_token:
            # Update session state with new tokens
            st.session_state.access_token = new_access_token
            st.session_state.refresh_token = new_refresh_token
            logger.debug("Updated tokens from response headers")
        
        if response.status_code == 401:
            # Try to refresh the token
            try:
                refresh_response = requests.post(
                    f"{BACKEND_URL}/refresh",
                    json={"refresh_token": st.session_state.refresh_token}
                )
                if refresh_response.status_code == 200:
                    data = refresh_response.json()
                    st.session_state.access_token = data["access_token"]
                    st.session_state.refresh_token = data["refresh_token"]
                    logger.debug("Successfully refreshed tokens")
                    # Retry the original request with new tokens
                    headers = {
                        "Authorization": f"Bearer {st.session_state.access_token}",
                        "X-Refresh-Token": st.session_state.refresh_token
                    }
                    kwargs["headers"] = headers
                    response = getattr(requests, method)(f"{BACKEND_URL}{endpoint}", **kwargs)
                else:
                    logger.error("Failed to refresh token")
                    st.error("Session expired. Please log in again.")
                    logout_user()
                    st.rerun()
            except Exception as e:
                logger.error(f"Token refresh failed: {str(e)}")
                st.error("Session expired. Please log in again.")
                logout_user()
                st.rerun()
                
        return response
    except Exception as e:
        logger.error(f"API request failed: {str(e)}")
        st.error(f"API request failed: {str(e)}")
        return None

# --- CASE MANAGEMENT ---
def get_wards():
    """Fetch available wards and cases."""
    try:
        response = make_authenticated_request("get", "/wards")
        if response and response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch wards: {response.json().get('error', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error fetching wards: {str(e)}")
        return None

def start_case(condition: str):
    """Start a new case with the selected condition."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/start_case",
            headers={"Authorization": f"Bearer {st.session_state.access_token}"},
            json={"condition": condition}
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.thread_id = data["thread_id"]
            st.session_state.case_started = True
            st.session_state.condition = condition
            st.session_state.case_variation = data["case_variation"]
            st.session_state.chat_history = [{"role": "assistant", "content": data["first_message"]}]
            return True
        else:
            st.error(f"Failed to start case: {response.json().get('error', 'Unknown error')}")
            return False
    except Exception as e:
        st.error(f"Error starting case: {str(e)}")
        return False

def continue_case(user_input: str):
    """Send user input to current case and get assistant's response."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/continue_case",
            json={
                "thread_id": st.session_state.thread_id,
                "user_input": user_input,
                "token": st.session_state.access_token,
                "refresh_token": st.session_state.refresh_token
            }
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.chat_history.extend([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": data["assistant_reply"]}
            ])
            
            if data.get("case_completed"):
                st.session_state.case_completed = True
                st.session_state.score = data.get("score")
                st.session_state.feedback = data.get("feedback")
                
                # Save performance data
                save_response = requests.post(
                    f"{BACKEND_URL}/save_performance",
                    json={
                        "thread_id": st.session_state.thread_id,
                        "score": data["score"],
                        "feedback": data["feedback"],
                        "token": st.session_state.access_token,
                        "refresh_token": st.session_state.refresh_token
                    }
                )
                
                if save_response.status_code == 200:
                    save_data = save_response.json()
                    if save_data.get("badge_awarded"):
                        st.toast(f"🏅 Congratulations! You earned the {save_data['badge_awarded']}!")
                else:
                    st.error(f"Failed to save performance: {save_response.json().get('error', 'Unknown error')}")
                
            return True
        else:
            st.error(f"Failed to continue case: {response.json().get('error', 'Unknown error')}")
            return False
    except Exception as e:
        st.error(f"Error continuing case: {str(e)}")
        return False

# --- UI COMPONENTS ---
def show_case_completion():
    """Display case completion UI."""
    st.success("🎉 Case completed!")
    
    # Display score and feedback
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Score", f"{st.session_state.score}/10")
    with col2:
        st.info(f"Feedback: {st.session_state.feedback}")
    
    # Display options
    st.markdown("### What would you like to do next?")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        if st.button("🔄 Try Another Case", key="try_another_case"):
            # Store the current condition
            current_condition = st.session_state.condition
            # Reset case state
            st.session_state.case_started = False
            st.session_state.case_completed = False
            st.session_state.thread_id = None
            st.session_state.chat_history = []
            # Start a fresh case
            start_case(current_condition)
            st.rerun()
    
    with col4:
        if st.button("📋 Choose Different Condition", key="choose_different"):
            # Reset all case state
            st.session_state.condition = None
            st.session_state.case_started = False
            st.session_state.case_completed = False
            st.session_state.thread_id = None
            st.session_state.chat_history = []
            st.rerun()
    
    with col5:
        if st.button("🚪 Logout", key="logout"):
            # Logout and clear session
            logout_user()
            st.rerun()

def show_chat_interface():
    """Display chat interface for ongoing cases."""
    st.title(f"📝 {st.session_state.condition} - Case {st.session_state.case_variation}")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Handle user input if case is not completed
    if not st.session_state.get("case_completed", False):
        user_input = st.chat_input("Your response...")
        if user_input:
            continue_case(user_input)
            st.rerun()
    else:
        show_case_completion()

def show_main_app():
    """Display main application interface."""
    st.sidebar.title("👤 User Info")
    st.sidebar.text(f"Email: {st.session_state.user['email']}")
    
    # Show progress statistics
    st.sidebar.title("📊 Progress")
    progress = make_authenticated_request("get", "/progress")
    if progress and progress.status_code == 200:
        stats = progress.json()["overall"]
        st.sidebar.metric("Cases Completed", stats["total_cases"])
        st.sidebar.metric("Average Score", f"{stats['average_score']:.1f}/10")
        st.sidebar.metric("Success Rate", f"{stats['success_rate']:.1f}%")
        st.sidebar.metric("Total Badges", stats["total_badges"])
    
    # Main content
    if not st.session_state.get("case_started", False):
        st.title("🏥 UKMLA Case Tutor")
        st.write("Select a ward and case to begin your training.")
        
        # Ward selection
        wards_data = get_wards()
        if wards_data:
            selected_ward = st.selectbox("Select Ward", list(CASES.keys()))
            if selected_ward:
                cases = [case["name"] for case in CASES[selected_ward]]
                selected_case = st.selectbox("Select Case", cases)
                if selected_case and st.button("Start Case"):
                    if start_case(selected_case):
                        st.rerun()
    else:
        show_chat_interface()

# --- CONSTANTS ---
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

# --- MAIN UI ---
def show_auth_page():
    """Display the authentication page."""
    st.title("🔐 UKMLA Case Tutor")
    
    # Toggle between login and signup
    auth_mode = st.radio("Choose an option:", ["Login", "Sign Up"])
    
    with st.form(key="auth_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Continue")

        if submit and email and password:
            if auth_mode == "Sign Up":
                result = signup_user(email, password)
                if result.get("success"):
                    st.success("✅ Account created! Please verify your email before logging in.")
                else:
                    st.error(f"❌ Signup failed: {result.get('error', 'Unknown error')}")
            else:
                result = login_user(email, password)
                if result.get("success"):
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error(f"❌ Login failed: {result.get('error', 'Invalid credentials')}")

# --- APP INITIALIZATION AND ROUTING ---
def main():
    """Main application entry point."""
    init_session_state()
    
    # Show error message if exists
    if st.session_state.error_message:
        st.error(st.session_state.error_message)
        if st.button("Clear Error"):
            st.session_state.error_message = None
            st.rerun()

    # Route to appropriate page based on auth status
    if not st.session_state.user:
        show_auth_page()
    else:
        show_main_app()

if __name__ == "__main__":
    main() 
