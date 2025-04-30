import time
from functools import wraps
from typing import Optional, Callable, Any, Dict
import uuid
from datetime import datetime, timezone

# In-memory storage for user sessions
user_sessions: Dict[str, Dict[str, Any]] = {}

def get_user_session_id(user_id: str) -> str:
    """
    Get or create a unique session ID for the current user.
    This ensures each user has their own isolated session.
    """
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            'session_id': str(uuid.uuid4()),
            'last_activity': datetime.now(timezone.utc),
            'state': {}
        }
    return user_sessions[user_id]['session_id']

def get_user_state(user_id: str, key: str, default: Any = None) -> Any:
    """
    Get a user-specific value from session state.
    """
    if user_id not in user_sessions:
        return default
    return user_sessions[user_id]['state'].get(key, default)

def set_user_state(user_id: str, key: str, value: Any):
    """
    Set a user-specific value in session state.
    """
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            'session_id': str(uuid.uuid4()),
            'last_activity': datetime.now(timezone.utc),
            'state': {}
        }
    user_sessions[user_id]['state'][key] = value
    user_sessions[user_id]['last_activity'] = datetime.now(timezone.utc)

def rate_limit(seconds: int = 1):
    """
    Decorator to prevent function from being called more frequently than specified.
    Uses in-memory storage to track last call time per user.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Optional[Any]:
            # Extract user_id from kwargs or first argument
            user_id = kwargs.get('user_id')
            if not user_id and args:
                user_id = args[0]
            
            if not user_id:
                return func(*args, **kwargs)
            
            # Get unique key for this function and user
            func_key = f"last_call_{func.__name__}"
            current_time = time.time()
            last_call = get_user_state(user_id, func_key, 0)
            
            # Check if enough time has passed
            if current_time - last_call < seconds:
                return None
                
            # Update last call time
            set_user_state(user_id, func_key, current_time)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def check_session_expiry(user_id: str) -> bool:
    """
    Check if the current session has expired due to inactivity.
    Returns True if session is valid, False if expired.
    """
    if user_id not in user_sessions:
        return False
        
    last_activity = user_sessions[user_id]['last_activity']
    idle_time = (datetime.now(timezone.utc) - last_activity).total_seconds()
    
    if idle_time > 3600:  # 1 hour timeout
        # Clear expired session
        del user_sessions[user_id]
        return False
        
    # Update last activity time
    user_sessions[user_id]['last_activity'] = datetime.now(timezone.utc)
    return True

def is_chat_ready(user_id: str) -> bool:
    """
    Check if chat is ready to accept input.
    Returns True if chat is ready, False otherwise.
    """
    if user_id not in user_sessions:
        return False
        
    case_started = get_user_state(user_id, 'case_started', False)
    thread_id = get_user_state(user_id, 'thread_id')
    is_loading = get_user_state(user_id, 'is_loading', False)
    
    return case_started and thread_id and not is_loading

def init_user_session(user_id: str):
    """
    Initialize or reset user-specific session state.
    Call this when user logs in or starts a new session.
    """
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            'session_id': str(uuid.uuid4()),
            'last_activity': datetime.now(timezone.utc),
            'state': {}
        }
    
    # Initialize user-specific state if not exists
    defaults = {
        'case_started': False,
        'thread_id': None,
        'chat_history': [],
        'condition': None,
        'is_loading': False,
        'case_completed': False,
        'score': None,
        'feedback': None,
        'case_variation': None,
        'total_score': 0
    }
    
    for key, default_value in defaults.items():
        if get_user_state(user_id, key) is None:
            set_user_state(user_id, key, default_value)

def clear_user_session(user_id: str):
    """
    Clear all user-specific session state.
    Call this on logout or session expiry.
    """
    if user_id in user_sessions:
        del user_sessions[user_id] 