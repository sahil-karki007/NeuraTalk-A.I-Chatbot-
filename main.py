import streamlit as st
import requests
import json
import sqlite3
import uuid
from datetime import datetime
from groq import Groq
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME = "llama-3.3-70b-versatile"  
DB_PATH     = "neuratalk.db"

# ─────────────────────────────────────────────
# PAGE SETUP  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NeuraTalk",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  — dark ChatGPT-style theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Hide Streamlit default chrome ── */
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1rem !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #12121f !important;
    border-right: 1px solid #2a2a45 !important;
}
section[data-testid="stSidebar"] * { color: #e8e8f0 !important; }

/* ── New Chat button ── */
div[data-testid="stSidebar"] .stButton > button {
    background: #6c63ff !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    width: 100% !important;
    padding: 10px !important;
    margin-bottom: 8px !important;
    box-shadow: 0 4px 14px rgba(108,99,255,0.35) !important;
    transition: all 0.2s !important;
}
div[data-testid="stSidebar"] .stButton > button:hover {
    background: #5a52e0 !important;
    transform: translateY(-1px) !important;
}

/* ── Main area ── */
.main { background: #0a0a14 !important; }

/* ── Message bubbles ── */
.user-msg {
    display: flex;
    justify-content: flex-end;
    margin: 10px 0;
    animation: slideUp 0.25s ease;
}
.user-bubble {
    background: #6c63ff;
    color: white;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    max-width: 65%;
    font-size: 15px;
    line-height: 1.6;
    box-shadow: 0 4px 14px rgba(108,99,255,0.3);
    word-wrap: break-word;
}

.ai-msg {
    display: flex;
    align-items: flex-end;
    gap: 10px;
    margin: 10px 0;
    animation: slideUp 0.25s ease;
}
.ai-avatar {
    font-size: 26px;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #1a1a2e;
    border-radius: 50%;
    border: 1px solid #2a2a45;
    flex-shrink: 0;
}
.ai-bubble {
    background: #1a1a2e;
    color: #e8e8f0;
    padding: 12px 18px;
    border-radius: 18px 18px 18px 4px;
    max-width: 65%;
    font-size: 15px;
    line-height: 1.6;
    border: 1px solid #2a2a45;
    word-wrap: break-word;
    white-space: pre-wrap;
}

/* ── Chat title in sidebar ── */
.chat-title-btn {
    background: transparent;
    border: none;
    color: #9999bb;
    text-align: left;
    width: 100%;
    padding: 8px 10px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 13px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    transition: background 0.15s;
}
.chat-title-btn:hover { background: #1f1f35; color: #e8e8f0; }
.chat-title-btn.active { background: #252540 !important; color: #e8e8f0 !important; }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    color: #6b6b8a;
    padding: 80px 20px;
}
.empty-state h2 { color: #9999bb; font-size: 24px; margin: 12px 0 8px; }

/* ── App title ── */
.app-title {
    font-size: 22px;
    font-weight: 700;
    color: #e8e8f0;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATABASE  — SQLite for chat memory
# ─────────────────────────────────────────────
def init_db():
    """Create tables if they don't exist yet."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Chats table — one row per conversation
    c.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            title TEXT DEFAULT 'New Chat',
            created_at TEXT
        )
    """)
    # Messages table — one row per message
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT,
            role TEXT,
            content TEXT,
            created_at TEXT,
            FOREIGN KEY (chat_id) REFERENCES chats(id)
        )
    """)
    conn.commit()
    conn.close()

def get_all_chats():
    """Return all chats ordered newest first."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, title FROM chats ORDER BY created_at DESC")
    chats = c.fetchall()   # list of (id, title) tuples
    conn.close()
    return chats

def create_chat():
    """Create a new empty chat and return its ID."""
    chat_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO chats (id, title, created_at) VALUES (?, ?, ?)",
        (chat_id, "New Chat", datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    return chat_id

def get_messages(chat_id):
    """Return all messages for a chat, oldest first."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT role, content FROM messages WHERE chat_id=? ORDER BY created_at ASC",
        (chat_id,)
    )
    messages = c.fetchall()   # list of (role, content) tuples
    conn.close()
    return messages

def save_message(chat_id, role, content):
    """Save one message to the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (chat_id, role, content, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def update_chat_title(chat_id, title):
    """Update the chat title (auto-generated from first message)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE chats SET title=? WHERE id=?", (title, chat_id))
    conn.commit()
    conn.close()

def delete_chat(chat_id):
    """Delete a chat and all its messages."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
    c.execute("DELETE FROM chats WHERE id=?", (chat_id,))
    conn.commit()
    conn.close()

def rename_chat(chat_id, new_title):
    """Rename a chat to a custom title."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE chats SET title=? WHERE id=?", (new_title, chat_id))
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# OLLAMA  — stream AI response
# ─────────────────────────────────────────────
def stream_ollama(messages_history):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages_history,
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    except Exception as e:
        yield f"❌ Error: {str(e)}"
# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────
init_db()   # create DB tables if first run

# Session state — persists across rerenders within same session
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None
if "renaming_chat_id" not in st.session_state:
    st.session_state.renaming_chat_id = None


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # App logo
    st.markdown('<div class="app-title">🤖 NeuraTalk</div>', unsafe_allow_html=True)

    # New Chat button
    if st.button("➕  New Chat", use_container_width=True):
        new_id = create_chat()
        st.session_state.active_chat_id = new_id
        st.session_state.renaming_chat_id = None
        st.rerun()   # refresh page to show new chat

    st.markdown("---")
    st.markdown("**💬 Chats**")

    # List all chats
    chats = get_all_chats()

    if not chats:
        st.caption("No chats yet. Start one above!")

    for chat_id, chat_title in chats:
        is_active = chat_id == st.session_state.active_chat_id

        col1, col2, col3 = st.columns([6, 1, 1])

        with col1:
            # Highlight active chat
            label = f"{'▶ ' if is_active else ''}{chat_title}"
            if st.button(label, key=f"chat_{chat_id}", use_container_width=True):
                st.session_state.active_chat_id = chat_id
                st.session_state.renaming_chat_id = None
                st.rerun()

        with col2:
            # Rename button
            if st.button("✏️", key=f"rename_{chat_id}", help="Rename"):
                st.session_state.renaming_chat_id = chat_id
                st.rerun()

        with col3:
            # Delete button
            if st.button("🗑️", key=f"del_{chat_id}", help="Delete"):
                delete_chat(chat_id)
                if st.session_state.active_chat_id == chat_id:
                    remaining = get_all_chats()
                    st.session_state.active_chat_id = remaining[0][0] if remaining else None
                st.rerun()

    # Rename input — shown when ✏️ clicked
    if st.session_state.renaming_chat_id:
        st.markdown("---")
        new_name = st.text_input(
            "New name:",
            key="rename_input",
            placeholder="Enter new chat name...",
        )
        col_save, col_cancel = st.columns(2)
        with col_save:
            if st.button("✅ Save", use_container_width=True):
                if new_name.strip():
                    rename_chat(st.session_state.renaming_chat_id, new_name.strip())
                st.session_state.renaming_chat_id = None
                st.rerun()
        with col_cancel:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.renaming_chat_id = None
                st.rerun()

    st.markdown("---")
    st.caption("Powered by Ollama 🦙")
    st.caption(f"Model: `{MODEL_NAME}`")


# ─────────────────────────────────────────────
# MAIN CHAT AREA
# ─────────────────────────────────────────────
chat_id = st.session_state.active_chat_id

if not chat_id:
    # No chat selected — show welcome screen
    st.markdown("""
    <div class="empty-state">
        <div style="font-size:60px">🤖</div>
        <h2>Welcome to NeuraTalk</h2>
        <p>Click <b>+ New Chat</b> in the sidebar to start a conversation</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Load messages for active chat
    messages = get_messages(chat_id)

    # Render all messages
    for role, content in messages:
        if role == "user":
            st.markdown(f"""
            <div class="user-msg">
                <div class="user-bubble">{content}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="ai-msg">
                <div class="ai-avatar">🤖</div>
                <div class="ai-bubble">{content}</div>
            </div>
            """, unsafe_allow_html=True)

    # Empty chat hint
    if not messages:
        st.markdown("""
        <div class="empty-state">
            <div style="font-size:48px">💬</div>
            <h2>New Conversation</h2>
            <p>Say something to get started!</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Input bar ──────────────────────────────
    user_input = st.chat_input("Message NeuraTalk...")

    if user_input:
        # Show user message immediately
        st.markdown(f"""
        <div class="user-msg">
            <div class="user-bubble">{user_input}</div>
        </div>
        """, unsafe_allow_html=True)

        # Save user message to DB
        save_message(chat_id, "user", user_input)

        # Auto-title the chat from first message
        if not messages:
            title = " ".join(user_input.split()[:5]) + "..."
            update_chat_title(chat_id, title)

        # Build history for Ollama (full conversation context)
        history = get_messages(chat_id)
        ollama_messages = [{"role": r, "content": c} for r, c in history]

        # Stream AI response into a placeholder
        with st.chat_message("assistant", avatar="🤖"):
            full_response = st.write_stream(stream_ollama(ollama_messages))

        # Save complete AI response to DB
        save_message(chat_id, "assistant", full_response)

        # Rerun to refresh sidebar titles
        st.rerun()