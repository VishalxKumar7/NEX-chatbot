import os
import re
import json
import subprocess
import time
import requests
import streamlit as st
import uuid

# ----------------------------
# Helper to sanitize question text for filenames
# ----------------------------
def sanitize_filename(text):
    filename = re.sub(r'[^a-zA-Z0-9_]+', '_', text.strip())
    return filename[:50] or "chat_default"

# ----------------------------
# Helper functions for chat persistence
# ----------------------------
def save_chat(chat_id):
    filename = f"chat_{chat_id}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(st.session_state.chat_history.get(chat_id, []), f)

def load_chat(chat_id):
    filename = f"chat_{chat_id}.json"
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# ----------------------------
# Function to start backend server automatically
# ----------------------------
def start_backend():
    try:
        requests.get("http://127.0.0.1:8000", timeout=2)
        print("✅ Backend already running.")
    except requests.exceptions.RequestException:
        print("🚀 Starting FastAPI backend...")
        subprocess.Popen(
            ["uvicorn", "backend.api:app", "--reload"],
            # Uncomment below lines to see backend logs during startup
            # stdout=subprocess.DEVNULL,
            # stderr=subprocess.STDOUT,
        )

        # Retry ping backend for up to 20 seconds
        for _ in range(20):
            try:
                requests.get("http://127.0.0.1:8000", timeout=2)
                print("✅ Backend started.")
                break
            except requests.exceptions.RequestException:
                time.sleep(1)
        else:
            print("❌ Backend failed to start in time.")

# Start backend on app launch
start_backend()

# ----------------------------
# Constants
# ----------------------------
BACKEND_URL = "http://127.0.0.1:8000"

# ----------------------------
# Streamlit app UI setup
# ----------------------------
st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title("📄 PDF RAG Chatbot")

# ----------------------------
# Initialize session state and load chat history
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = "new_chat"
    st.session_state.chat_history[st.session_state.current_chat] = []

else:
    if st.session_state.current_chat not in st.session_state.chat_history:
        st.session_state.chat_history[st.session_state.current_chat] = load_chat(st.session_state.current_chat)

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.title("Controls")

if st.sidebar.button("New Chat"):
    st.session_state.current_chat = "new_chat"
    st.session_state.chat_history[st.session_state.current_chat] = []

chat_ids = list(st.session_state.chat_history.keys())

selected_chat = st.sidebar.selectbox(
    "Chat History",
    chat_ids,
    index=chat_ids.index(st.session_state.current_chat) if st.session_state.current_chat in chat_ids else 0
)

if selected_chat not in st.session_state.chat_history:
    st.session_state.chat_history[selected_chat] = load_chat(selected_chat)
st.session_state.current_chat = selected_chat

if st.sidebar.button("Delete Chat"):
    if st.session_state.current_chat in st.session_state.chat_history:
        filename = f"chat_{st.session_state.current_chat}.json"
        if os.path.exists(filename):
            os.remove(filename)
        del st.session_state.chat_history[st.session_state.current_chat]
        st.session_state.current_chat = "new_chat"
        st.session_state.chat_history[st.session_state.current_chat] = []

if st.sidebar.button("Download Chat"):
    chat = st.session_state.chat_history.get(st.session_state.current_chat, [])
    if chat:
        filename = f"chat_{st.session_state.current_chat}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            for msg in chat:
                f.write(f"You: {msg['question']}\nBot: {msg['answer']}\n\n")
        st.sidebar.success(f"Chat saved as {filename}")

# ----------------------------
# Display chat messages
# ----------------------------
def display_chat():
    chat = st.session_state.chat_history.get(st.session_state.current_chat, [])
    for msg in chat:
        st.markdown(f"**You:** {msg['question']}")
        st.markdown(f"**Bot:** {msg['answer']}")

display_chat()

# ----------------------------
# Input form for user questions
# ----------------------------
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your question here...")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():

    # If new chat, rename current_chat to sanitized first question
    if st.session_state.current_chat == "new_chat" or st.session_state.current_chat.startswith("new_chat"):
        new_chat_id = sanitize_filename(user_input)
        st.session_state.current_chat = new_chat_id
        if new_chat_id not in st.session_state.chat_history:
            st.session_state.chat_history[new_chat_id] = load_chat(new_chat_id)

    if st.session_state.current_chat not in st.session_state.chat_history:
        st.session_state.chat_history[st.session_state.current_chat] = []

    placeholder = st.empty()
    with placeholder.container():
        st.markdown("**Bot is typing...** :hourglass_flowing_sand:")

    try:
        payload = {
            "chat_id": str(st.session_state.current_chat),
            "question": str(user_input)
        }
        response = requests.post(f"{BACKEND_URL}/ask", json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        st.session_state.chat_history[st.session_state.current_chat] = data.get("history", [])
        save_chat(st.session_state.current_chat)
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with backend: {e}")

    placeholder.empty()
    display_chat()
