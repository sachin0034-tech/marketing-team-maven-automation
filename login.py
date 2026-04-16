import streamlit as st
import os
import hashlib
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Credential resolution — dev: .env file  |  prod: Streamlit secrets / env vars
# ---------------------------------------------------------------------------

def _get_secret(key: str) -> str:
    """Read from st.secrets first (Streamlit Cloud), then fall back to env vars."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        value = os.getenv(key)
        if not value:
            raise RuntimeError(
                f"Missing required secret '{key}'. "
                "Set it in .env (local) or Streamlit secrets (production)."
            )
        return value


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def check_credentials(email: str, password: str) -> bool:
    expected_email = _get_secret("APP_EMAIL")
    expected_password_hash = _get_secret("APP_PASSWORD_HASH")
    return email == expected_email and _hash(password) == expected_password_hash


# ---------------------------------------------------------------------------
# Login UI
# ---------------------------------------------------------------------------

def show_login():
    st.set_page_config(page_title="Login", page_icon="🔒", layout="centered")

    st.markdown(
        """
        <style>
        .login-box {
            max-width: 420px;
            margin: auto;
            padding: 2.5rem 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown("<div class='login-box'>", unsafe_allow_html=True)
        st.title("Sign in")
        st.caption("Marketing Dashboard")

        with st.form("login_form"):
            email = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)

        if submitted:
            if not email or not password:
                st.error("Email and password are required.")
            elif check_credentials(email, password):
                st.session_state["authenticated"] = True
                st.session_state["user_email"] = email
                st.rerun()
            else:
                st.error("Invalid email or password.")

        st.markdown("</div>", unsafe_allow_html=True)


def require_login():
    """Call this at the top of every page to gate access."""
    if not st.session_state.get("authenticated"):
        show_login()
        st.stop()
