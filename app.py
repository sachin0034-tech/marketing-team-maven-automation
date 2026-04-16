import streamlit as st
from login import require_login

st.set_page_config(page_title="Marketing Dashboard", page_icon="📊", layout="wide")

require_login()

# ---------------------------------------------------------------------------
# Sidebar — API key + navigation + sign out
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        value=st.session_state.get("openai_api_key", ""),
        help="Your key is stored only in this browser session and never saved.",
    )
    if api_key:
        st.session_state["openai_api_key"] = api_key
        st.success("API key saved for this session.", icon="✅")

    st.divider()

    st.markdown("## 🗂️ Tools")
    page = st.radio(
        "Select a tool",
        options=["🏠 Home", "📞 Sales Call Analyzer", "🎓 Student Match", "📋 LinkedIn Review"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(f"**{st.session_state.get('user_email', '')}**")
    if st.button("Sign out", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------

if page == "🏠 Home":
    st.title("📊 Marketing Dashboard")
    st.markdown(
        """
        Welcome! Use the sidebar to navigate between tools.

        | Tool | Description |
        |------|-------------|
        | **Sales Call Analyzer** | Upload a call transcript or paste a URL to get AI coaching and a ready-made 15-min pitch |
        | **Student Match** | Upload enrolled & dropped-off sheets to find students that appear in both |
        | **LinkedIn Review** | Paste a LinkedIn post and get a full 6-layer / 28-checkpoint audit with per-layer scores, specific suggestions for every failing checkpoint, and prioritised improvement ideas |

        > Make sure you've entered your **OpenAI API key** in the sidebar before using any tool.
        """
    )

elif page == "📞 Sales Call Analyzer":
    from pages.transcript_analyzer import render
    render()

elif page == "🎓 Student Match":
    from pages.student_match import render
    render()

elif page == "📋 LinkedIn Review":
    from pages.review import render
    render()