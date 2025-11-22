<<<<<<< HEAD
import streamlit as st
from rag_core import rag_answer
import uuid
from datetime import datetime

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Dementia RAG Assistant",
    page_icon="🧠",
    layout="wide",
)


# -------------------------------------------------
# INITIAL SESSION STATE
# -------------------------------------------------
if "conversations" not in st.session_state:
    st.session_state.conversations = {}  # id -> {title, created_at, messages}

if "current_conv_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.current_conv_id = new_id
    st.session_state.conversations[new_id] = {
        "title": "New chat",
        "created_at": datetime.now(),
        "messages": [
            {
                "role": "assistant",
                "content": "Hello! I’m your dementia-care assistant. Ask me anything about dementia.",
            }
        ],
    }

# Text that pre-fills the input box (used for Copy/Edit)
if "input_draft" not in st.session_state:
    st.session_state.input_draft = ""

# Dark / light mode
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 Dementia Assistant")

    # Dark mode toggle
    st.session_state.dark_mode = st.toggle("🌙 Dark mode", value=st.session_state.dark_mode)

    st.markdown("---")

    # New chat
    if st.button("➕ New chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.current_conv_id = new_id
        st.session_state.conversations[new_id] = {
            "title": "New chat",
            "created_at": datetime.now(),
            "messages": [
                {
                    "role": "assistant",
                    "content": "New chat started. How can I help you today?",
                }
            ],
        }
        st.session_state.input_draft = ""
        st.rerun()

    st.markdown("### 📚 Past chats")

    # List chats (most recent first)
    sorted_chats = sorted(
        st.session_state.conversations.items(),
        key=lambda kv: kv[1]["created_at"],
        reverse=True,
    )

    for cid, conv in sorted_chats:
        label = conv["title"] if conv["title"] != "New chat" else f"Chat {cid[:8]}"
        is_current = cid == st.session_state.current_conv_id
        style = "font-weight: 600;" if is_current else ""

        if st.button(label, key=f"conv_{cid}", use_container_width=True):
            st.session_state.current_conv_id = cid
            st.session_state.input_draft = ""
            st.rerun()

    st.markdown("---")

    # Delete current chat
    if len(st.session_state.conversations) > 1:
        if st.button("🗑 Delete current chat", type="secondary", use_container_width=True):
            cur = st.session_state.current_conv_id
            del st.session_state.conversations[cur]
            # switch to another chat
            st.session_state.current_conv_id = list(st.session_state.conversations.keys())[0]
            st.session_state.input_draft = ""
            st.rerun()


# -------------------------------------------------
# THEME STYLES
# -------------------------------------------------
if st.session_state.dark_mode:
    bg_main = "#0f172a"
    bg_card = "#111827"
    bg_user = "#1e293b"
    bg_bot = "#022c22"
    text_color = "#e5e7eb"
else:
    bg_main = "#f5f5f5"
    bg_card = "#ffffff"
    bg_user = "#e5f0ff"
    bg_bot = "#fff7d1"
    text_color = "#111827"

st.markdown(
    f"""
<style>
body {{
    background-color: {bg_main};
    color: {text_color};
}}

.main-block {{
    background-color: {bg_card};
    padding: 1.5rem;
    border-radius: 1.5rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}}

.chat-container {{
    max-height: 65vh;
    overflow-y: auto;
    padding-right: 0.5rem;
}}

.user-bubble {{
    background-color: {bg_user};
    padding: 0.75rem 1rem;
    border-radius: 18px;
    margin-bottom: 0.5rem;
}}

.bot-bubble {{
    background-color: {bg_bot};
    padding: 0.75rem 1rem;
    border-radius: 18px;
    margin-bottom: 0.5rem;
}}

.msg-meta {{
    font-size: 0.75rem;
    opacity: 0.7;
    margin-bottom: 0.1rem;
}}
.inline-btn {{
    font-size: 0.8rem;
    margin-right: 0.4rem;
}}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------
# MAIN CHAT AREA
# -------------------------------------------------
conv = st.session_state.conversations[st.session_state.current_conv_id]
messages = conv["messages"]

st.markdown("### 💬 Dementia Care Chat")
st.caption("Ask questions, view previous answers, and refine responses.")

with st.container():
    st.markdown('<div class="main-block">', unsafe_allow_html=True)

    # Scrollable chat area
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for idx, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]

        with st.chat_message("user" if role == "user" else "assistant"):
            if role == "user":
                st.markdown(f"<div class='user-bubble'>{content}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='bot-bubble'>{content}</div>", unsafe_allow_html=True)

                # Action buttons for assistant messages
                bcol1, bcol2, bcol3 = st.columns([1, 1, 4])
                with bcol1:
                    if st.button("📋 Copy/Edit", key=f"copy_{idx}", help="Copy this answer to input box"):
                        st.session_state.input_draft = content
                        st.rerun()
                with bcol2:
                    # regenerate based on the previous user message
                    if st.button("🔁 Regenerate", key=f"regen_{idx}", help="Regenerate this answer"):
                        # find last user message before this assistant msg
                        last_user = None
                        for j in range(idx - 1, -1, -1):
                            if messages[j]["role"] == "user":
                                last_user = messages[j]["content"]
                                break
                        if last_user:
                            # remove this assistant msg
                            messages.pop(idx)
                            # call model again
                            with st.spinner("Regenerating..."):
                                new_reply = rag_answer(last_user, messages)
                            messages.append({"role": "assistant", "content": new_reply})
                            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)  # end chat-container

    st.markdown("---")

    # -------------------------------------------------
    # INPUT AREA
    # -------------------------------------------------
    user_input = st.text_area(
        "Type your message:",
        value=st.session_state.input_draft,
        key="chat_input",
        height=90,
    )

    icol1, icol2, icol3 = st.columns([5, 1.5, 1.5])

    with icol1:
        if st.button("Send ▶", type="primary", use_container_width=True):
            text = user_input.strip()
            if text:
                # first user message becomes chat title
                if conv["title"] == "New chat":
                    conv["title"] = text[:40] + ("…" if len(text) > 40 else "")

                messages.append({"role": "user", "content": text})

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        reply = rag_answer(text, messages)

                messages.append({"role": "assistant", "content": reply})

                st.session_state.input_draft = ""
                st.rerun()

    with icol2:
        if st.button("✏️ Edit last", use_container_width=True):
            # find last user message
            last_user = None
            for m in reversed(messages):
                if m["role"] == "user":
                    last_user = m["content"]
                    break
            if last_user:
                st.session_state.input_draft = last_user
                st.rerun()

    with icol3:
        if st.button("🧹 Clear", use_container_width=True):
            st.session_state.input_draft = ""
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)  # end main-block

# Save back to session_state (not strictly necessary since we mutate in place,
# but good for clarity)
st.session_state.conversations[st.session_state.current_conv_id]["messages"] = messages
=======
import streamlit as st
from rag_core import rag_answer
import uuid
from datetime import datetime

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Dementia RAG Assistant",
    page_icon="🧠",
    layout="wide",
)


# -------------------------------------------------
# INITIAL SESSION STATE
# -------------------------------------------------
if "conversations" not in st.session_state:
    st.session_state.conversations = {}  # id -> {title, created_at, messages}

if "current_conv_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.current_conv_id = new_id
    st.session_state.conversations[new_id] = {
        "title": "New chat",
        "created_at": datetime.now(),
        "messages": [
            {
                "role": "assistant",
                "content": "Hello! I’m your dementia-care assistant. Ask me anything about dementia.",
            }
        ],
    }

# Text that pre-fills the input box (used for Copy/Edit)
if "input_draft" not in st.session_state:
    st.session_state.input_draft = ""

# Dark / light mode
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 Dementia Assistant")

    # Dark mode toggle
    st.session_state.dark_mode = st.toggle("🌙 Dark mode", value=st.session_state.dark_mode)

    st.markdown("---")

    # New chat
    if st.button("➕ New chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.current_conv_id = new_id
        st.session_state.conversations[new_id] = {
            "title": "New chat",
            "created_at": datetime.now(),
            "messages": [
                {
                    "role": "assistant",
                    "content": "New chat started. How can I help you today?",
                }
            ],
        }
        st.session_state.input_draft = ""
        st.rerun()

    st.markdown("### 📚 Past chats")

    # List chats (most recent first)
    sorted_chats = sorted(
        st.session_state.conversations.items(),
        key=lambda kv: kv[1]["created_at"],
        reverse=True,
    )

    for cid, conv in sorted_chats:
        label = conv["title"] if conv["title"] != "New chat" else f"Chat {cid[:8]}"
        is_current = cid == st.session_state.current_conv_id
        style = "font-weight: 600;" if is_current else ""

        if st.button(label, key=f"conv_{cid}", use_container_width=True):
            st.session_state.current_conv_id = cid
            st.session_state.input_draft = ""
            st.rerun()

    st.markdown("---")

    # Delete current chat
    if len(st.session_state.conversations) > 1:
        if st.button("🗑 Delete current chat", type="secondary", use_container_width=True):
            cur = st.session_state.current_conv_id
            del st.session_state.conversations[cur]
            # switch to another chat
            st.session_state.current_conv_id = list(st.session_state.conversations.keys())[0]
            st.session_state.input_draft = ""
            st.rerun()


# -------------------------------------------------
# THEME STYLES
# -------------------------------------------------
if st.session_state.dark_mode:
    bg_main = "#0f172a"
    bg_card = "#111827"
    bg_user = "#1e293b"
    bg_bot = "#022c22"
    text_color = "#e5e7eb"
else:
    bg_main = "#f5f5f5"
    bg_card = "#ffffff"
    bg_user = "#e5f0ff"
    bg_bot = "#fff7d1"
    text_color = "#111827"

st.markdown(
    f"""
<style>
body {{
    background-color: {bg_main};
    color: {text_color};
}}

.main-block {{
    background-color: {bg_card};
    padding: 1.5rem;
    border-radius: 1.5rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}}

.chat-container {{
    max-height: 65vh;
    overflow-y: auto;
    padding-right: 0.5rem;
}}

.user-bubble {{
    background-color: {bg_user};
    padding: 0.75rem 1rem;
    border-radius: 18px;
    margin-bottom: 0.5rem;
}}

.bot-bubble {{
    background-color: {bg_bot};
    padding: 0.75rem 1rem;
    border-radius: 18px;
    margin-bottom: 0.5rem;
}}

.msg-meta {{
    font-size: 0.75rem;
    opacity: 0.7;
    margin-bottom: 0.1rem;
}}
.inline-btn {{
    font-size: 0.8rem;
    margin-right: 0.4rem;
}}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------
# MAIN CHAT AREA
# -------------------------------------------------
conv = st.session_state.conversations[st.session_state.current_conv_id]
messages = conv["messages"]

st.markdown("### 💬 Dementia Care Chat")
st.caption("Ask questions, view previous answers, and refine responses.")

with st.container():
    st.markdown('<div class="main-block">', unsafe_allow_html=True)

    # Scrollable chat area
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for idx, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]

        with st.chat_message("user" if role == "user" else "assistant"):
            if role == "user":
                st.markdown(f"<div class='user-bubble'>{content}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='bot-bubble'>{content}</div>", unsafe_allow_html=True)

                # Action buttons for assistant messages
                bcol1, bcol2, bcol3 = st.columns([1, 1, 4])
                with bcol1:
                    if st.button("📋 Copy/Edit", key=f"copy_{idx}", help="Copy this answer to input box"):
                        st.session_state.input_draft = content
                        st.rerun()
                with bcol2:
                    # regenerate based on the previous user message
                    if st.button("🔁 Regenerate", key=f"regen_{idx}", help="Regenerate this answer"):
                        # find last user message before this assistant msg
                        last_user = None
                        for j in range(idx - 1, -1, -1):
                            if messages[j]["role"] == "user":
                                last_user = messages[j]["content"]
                                break
                        if last_user:
                            # remove this assistant msg
                            messages.pop(idx)
                            # call model again
                            with st.spinner("Regenerating..."):
                                new_reply = rag_answer(last_user, messages)
                            messages.append({"role": "assistant", "content": new_reply})
                            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)  # end chat-container

    st.markdown("---")

    # -------------------------------------------------
    # INPUT AREA
    # -------------------------------------------------
    user_input = st.text_area(
        "Type your message:",
        value=st.session_state.input_draft,
        key="chat_input",
        height=90,
    )

    icol1, icol2, icol3 = st.columns([5, 1.5, 1.5])

    with icol1:
        if st.button("Send ▶", type="primary", use_container_width=True):
            text = user_input.strip()
            if text:
                # first user message becomes chat title
                if conv["title"] == "New chat":
                    conv["title"] = text[:40] + ("…" if len(text) > 40 else "")

                messages.append({"role": "user", "content": text})

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        reply = rag_answer(text, messages)

                messages.append({"role": "assistant", "content": reply})

                st.session_state.input_draft = ""
                st.rerun()

    with icol2:
        if st.button("✏️ Edit last", use_container_width=True):
            # find last user message
            last_user = None
            for m in reversed(messages):
                if m["role"] == "user":
                    last_user = m["content"]
                    break
            if last_user:
                st.session_state.input_draft = last_user
                st.rerun()

    with icol3:
        if st.button("🧹 Clear", use_container_width=True):
            st.session_state.input_draft = ""
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)  # end main-block

# Save back to session_state (not strictly necessary since we mutate in place,
# but good for clarity)
st.session_state.conversations[st.session_state.current_conv_id]["messages"] = messages
>>>>>>> da25d05 (Initial commit of dementia RAG app)
