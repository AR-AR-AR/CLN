"""
Authentication and session management for Tastytrade API
"""

from tastytrade import Session
import traceback


class AuthManager:
    """Handles authentication and session management for Tastytrade"""

    def __init__(self):
        self._session = None
        self._credentials = {}

    def login(self, email, password):
        """
        Authenticate with Tastytrade API

        Args:
            email (str): User email
            password (str): User password

        Returns:
            dict: Authentication result with success status and message
        """
        try:
            session = Session(email, password)

            # Store session and credentials
            self._session = session
            self._credentials = {
                "email": email,
                "password": password,
                "logged_in": True
            }

            return {
                "success": True,
                "message": "✅ Login successful!",
                "color": "success",
                "data": self._credentials
            }

        except Exception as e:
            print(f"❌ Login failed: {e}")
            traceback.print_exc()

            # Clear any stored data on failure
            self._session = None
            self._credentials = {}

            return {
                "success": False,
                "message": f"❌ Login failed: {e}",
                "color": "danger",
                "data": {}
            }

    def get_session(self, email=None, password=None):
        """
        Get current session or create new one with provided credentials

        Args:
            email (str, optional): Email for new session
            password (str, optional): Password for new session

        Returns:
            Session: Tastytrade session object
        """
        if email and password:
            # Create new session with provided credentials
            return Session(email, password)
        elif self._session:
            # Return stored session
            return self._session
        else:
            raise ValueError("No active session and no credentials provided")

    def is_authenticated(self):
        """Check if user is currently authenticated"""
        return self._credentials.get('logged_in', False)

    def get_credentials(self):
        """Get stored credentials"""
        return self._credentials.copy()

    def logout(self):
        """Clear session and credentials"""
        self._session = None
        self._credentials = {}


# Global auth manager instance
auth_manager = AuthManager()


def validate_session_data(session_data):
    """
    Validate session data from Dash store

    Args:
        session_data (dict): Session data from Dash store

    Returns:
        bool: True if session data is valid and user is logged in
    """
    if not session_data:
        return False

    return session_data.get('logged_in', False)


def create_session_from_data(session_data):
    """
    Create Tastytrade session from stored session data

    Args:
        session_data (dict): Session data containing credentials

    Returns:
        Session: Tastytrade session object

    Raises:
        ValueError: If session data is invalid
    """
    if not validate_session_data(session_data):
        raise ValueError("Invalid or missing session data")

    email = session_data.get('email')
    password = session_data.get('password')

    if not email or not password:
        raise ValueError("Missing email or password in session data")

    return Session(email, password)