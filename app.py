import streamlit as st
import google.generativeai as genai
import os
import uuid
from datetime import datetime
from supabase import create_client, Client
from streamlit_cookies_controller import CookieController

# --- 1. Page Configuration ---
st.set_page_config(page_title="Dadi AI", page_icon="👵🏾", layout="centered")

def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name, encoding="utf-8") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def load_html(file_name):
    if os.path.exists(file_name):
        with open(file_name, encoding="utf-8") as f:
            st.markdown(f.read(), unsafe_allow_html=True)

load_css("style.css")
load_html("header.html")

# --- 2. Identity Management (Cookies) ---
# Initialize the cookie controller
cookie_controller = CookieController()

# Check if the user already has an ID cookie
current_user_id = cookie_controller.get('dadi_user_id')

# If it's a new user, generate an ID and save it to their browser
if not current_user_id:
    current_user_id = str(uuid.uuid4())
    # Save the cookie for 1 year (31536000 seconds)
    cookie_controller.set('dadi_user_id', current_user_id, max_age=31536000)

# --- 3. Connections Setup (Gemini & Supabase) ---
api_key = st.secrets.get("GEMINI_API_KEY")
supa_url = st.secrets.get("SUPABASE_URL")
supa_key = st.secrets.get("SUPABASE_KEY")

if not all([api_key, supa_url, supa_key]):
    st.error("Arre beta! Check your secrets.toml. Something is missing.")
    st.stop()

genai.configure(api_key=api_key)
DADI_PROMPT = """
You are Dadi, a typical Indian grandmother. You are fun, a bit naughty, and love to roast people (especially the younger generation) about their lifestyle, waking up late, eating junk food, or being glued to their phones. 
However, beneath the roasting, you are incredibly kind, wise, and deeply care for them. After roasting them, you must always offer genuine comfort and practical, traditional wisdom to fix their problems. 
Use common Hindi slang naturally (like beta, arre, nalayak, shabash, hai ram) but keep the main language English. Be witty, sarcastic, but ultimately loving and helpful. Keep responses formatted clearly.
"""
model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=DADI_PROMPT)
supabase: Client = create_client(supa_url, supa_key)

# --- 4. Database Sync (Now with user_id!) ---
def fetch_chats_from_db(user_id):
    """Pulls ONLY this specific user's conversations from PostgreSQL."""
    try:
        # Added .eq("user_id", user_id) to filter the results!
        response = supabase.table("dadi_chats").select("*").eq("user_id", user_id).order("last_updated", desc=True).execute()
        history = {}
        for row in response.data:
            history[row["session_id"]] = {
                "title": row["title"],
                "messages": row["messages"],
                "gemini_history": row["gemini_history"]
            }
        return history
    except Exception as e:
        st.error(f"Database error: {e}")
        return {}

def save_chat_to_db(session_id, user_id, chat_data):
    """Upserts the current conversation and tags it with the user_id."""
    data_to_insert = {
        "session_id": session_id,
        "user_id": user_id,  # Link this chat to the user's cookie ID
        "title": chat_data["title"],
        "messages": chat_data["messages"],
        "gemini_history": chat_data["gemini_history"],
        "last_updated": datetime.now().isoformat()
    }
    supabase.table("dadi_chats").upsert(data_to_insert).execute()

# --- 5. Initialize State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = fetch_chats_from_db(current_user_id)

if not st.session_state.chat_history:
    initial_id = str(uuid.uuid4())
    st.session_state.current_session_id = initial_id
    st.session_state.chat_history[initial_id] = {
        "title": f"Chat ({datetime.now().strftime('%d %b, %H:%M')})",
        "messages": [{"role": "assistant", "content": "Arre beta! Finally you have time for your Dadi. Come, sit. Tell me, what trouble have you gotten yourself into today?"}],
        "gemini_history": []
    }
    save_chat_to_db(initial_id, current_user_id, st.session_state.chat_history[initial_id])
elif "current_session_id" not in st.session_state:
    st.session_state.current_session_id = list(st.session_state.chat_history.keys())[0]

# --- 6. Sidebar Navigation ---
with st.sidebar:
    st.title("🕰️ Dadi's Memory")
    st.caption(f"ID: {current_user_id[:8]}...") # Small visual indicator of their ID
    
    if st.button("➕ New Conversation", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.current_session_id = new_id
        st.session_state.chat_history[new_id] = {
            "title": f"Chat ({datetime.now().strftime('%d %b, %H:%M')})",
            "messages": [{"role": "assistant", "content": "Arre beta! What new problem have you brought me today?"}],
            "gemini_history": []
        }
        save_chat_to_db(new_id, current_user_id, st.session_state.chat_history[new_id])
        st.rerun()
        
    st.divider()
    
    for session_id, session_data in st.session_state.chat_history.items():
        button_type = "primary" if session_id == st.session_state.current_session_id else "secondary"
        if st.button(session_data["title"], key=f"btn_{session_id}", use_container_width=True, type=button_type):
            st.session_state.current_session_id = session_id
            st.rerun()

# --- 7. Active Chat Logic ---
current_chat = st.session_state.chat_history[st.session_state.current_session_id]

formatted_history = [{"role": msg["role"], "parts": [msg["parts"][0]]} for msg in current_chat["gemini_history"]] if current_chat["gemini_history"] else []
chat_session = model.start_chat(history=formatted_history)

for msg in current_chat["messages"]:
    avatar_icon = "👵🏾" if msg["role"] == "assistant" else "🧑🏽"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        st.markdown(msg["content"])

if user_input := st.chat_input("Tell Dadi your problems..."):
    current_chat["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑🏽"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="👵🏾"):
        message_placeholder = st.empty()
        message_placeholder.markdown("*(Dadi is adjusting her glasses and typing...)*")
        
        try:
            response = chat_session.send_message(user_input)
            message_placeholder.markdown(response.text)
            
            current_chat["messages"].append({"role": "assistant", "content": response.text})
            
            clean_history = [{"role": m.role, "parts": [p.text for p in m.parts]} for m in chat_session.history]
            current_chat["gemini_history"] = clean_history
            
            # Save using the user's specific ID
            save_chat_to_db(st.session_state.current_session_id, current_user_id, current_chat)
            
        except Exception as e:
            error_msg = f"Arre, my internet is hanging! (Error: {e})"
            message_placeholder.error(error_msg)