import streamlit as st
import os
import time
import traceback
from openai import OpenAI
from typing import List
from dataclasses import dataclass
from typing import Literal
from streamlit_float import *
import re

# --- Configuration ---
ASSISTANT_NAME = "PDF_QA_Assistant"
ASSISTANT_MODEL = "gpt-4o-mini"
ASSISTANT_INSTRUCTIONS = (
    "You are a helpful assistant. Your primary function is to answer questions "
    "based *solely* on the content of the PDF files provided in the current thread. "
    "Do not use any external knowledge or information outside of these files. "
    "If the answer cannot be found within the provided files, explicitly state: "
    "'I can't help with that.' "
)

POLLING_INTERVAL_S = 3
PDF_FILES = [
    "docs/new_draft_bot_25-03-25.pdf",
    "docs/graphs2.pdf",
    "docs/LIWC_vars.pdf",
    "docs/website_description.pdf",
    "docs/QnA_chatbot.pdf",
]

# Initialize OpenAI client
API_KEY = None
try:
    API_KEY = st.secrets["OPENAI_API_KEY"]
    print("Using API Key from Streamlit secrets.")
except (KeyError, FileNotFoundError):
    print("Streamlit secrets not found or key 'OPENAI_API_KEY' missing.")
    print("Attempting to use environment variable 'OPENAI_API_KEY'...")
    API_KEY = os.environ.get("OPENAI_API_KEY")
    if API_KEY:
        print("Using API Key from environment variable.")
    else:
        st.error(
            "OpenAI API Key not found. Please set it in Streamlit secrets (secrets.toml) or as an environment variable OPENAI_API_KEY."
        )
        # Don't st.stop() here, let the UI show the error and potentially fail later if needed
        # st.stop()

# --- Initialize OpenAI Client ---
client = None
if API_KEY:
    try:
        client = OpenAI(api_key=API_KEY)
        print("OpenAI client initialized successfully.")
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        # client remains None
else:
    # Error already shown above if key is missing
    pass

# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# --- Core Chatbot Functions ---
def upload_files(file_paths: List[str]) -> List[str]:
    """Uploads multiple files to OpenAI and returns their IDs."""
    uploaded_file_ids = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            continue
        try:
            with open(file_path, "rb") as f:
                response = client.files.create(file=f, purpose="assistants")
                uploaded_file_ids.append(response.id)
        except Exception as e:
            print(f"Error uploading file {file_path}: {e}")
    return uploaded_file_ids


def wait_for_run_completion(thread_id: str, run_id: str) -> str:
    """Polls the Run status until it's completed or fails."""
    while True:
        try:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread_id, run_id=run_id
            )
            if run_status.status == "completed":
                return "completed"
            elif run_status.status in ["queued", "in_progress", "requires_action"]:
                time.sleep(POLLING_INTERVAL_S)
            else:
                return run_status.status
        except Exception as e:
            time.sleep(POLLING_INTERVAL_S * 2)


def get_assistant_response_text(thread_id: str) -> str:
    """Retrieves and processes the assistant message from the thread."""
    try:
        messages = client.beta.threads.messages.list(
            thread_id=thread_id, order="desc", limit=1
        )
        for msg in messages.data:
            if msg.role == "assistant":
                for content_block in msg.content:
                    if content_block.type == "text":
                        text_content = content_block.text
                        # Remove citation markers using annotations
                        clean_text = text_content.value
                        # Process annotations in reverse order
                        for annotation in reversed(text_content.annotations):
                            clean_text = (
                                clean_text[: annotation.start_index]
                                + annotation.text
                                + clean_text[annotation.end_index :]
                            )

                        # Additional cleanup for any remaining 【】 patterns
                        clean_text = re.sub(r"【.*?】", "", clean_text)

                        return clean_text.strip()
        return "No response found from the assistant."
    except Exception as e:
        print(f"Error retrieving messages: {e}")
        return "Error retrieving response."


def initialize_assistant():
    """Initialize assistant and files once per session"""
    if "assistant_initialized" not in st.session_state:
        # Upload files
        st.session_state.uploaded_file_ids = upload_files(PDF_FILES)

        # Create assistant
        assistant = client.beta.assistants.create(
            name=ASSISTANT_NAME,
            instructions=ASSISTANT_INSTRUCTIONS,
            model=ASSISTANT_MODEL,
            tools=[{"type": "file_search"}],
        )
        st.session_state.assistant_id = assistant.id
        st.session_state.assistant_initialized = True


def process_user_query(question: str) -> str:
    """Process user query using OpenAI Assistants API"""
    # Create thread if not exists
    if "thread_id" not in st.session_state:
        thread = client.beta.threads.create()
        st.session_state.thread_id = thread.id

    # Add message to thread
    message_attachments = [
        {"file_id": file_id, "tools": [{"type": "file_search"}]}
        for file_id in st.session_state.uploaded_file_ids
    ]
    client.beta.threads.messages.create(
        thread_id=st.session_state.thread_id,
        role="user",
        content=question,
        attachments=message_attachments,
    )

    # Create and run
    run = client.beta.threads.runs.create(
        thread_id=st.session_state.thread_id,
        assistant_id=st.session_state.assistant_id,
    )

    # Wait for completion
    run_status = wait_for_run_completion(st.session_state.thread_id, run.id)

    if run_status == "completed":
        return get_assistant_response_text(st.session_state.thread_id)
    return "Error processing your request."


def is_graph_related(query: str) -> bool:
    """
    Check if the query is related to a graph using enhanced keyword matching.
    """
    graph_keywords = {
        "graph",
        "plot",
        "figure",
        "chart",
        "diagram",
        "line",
        "bar",
        "scatter",
        "table",
        "violin",
        "box",
        "trend",
        "curve",
        "axis",
        "data point",
        "visualization",
        "r2",
        "model performance",
        "distribution",
    }
    query_lower = query.lower().strip()
    words = query_lower.split()

    if any(keyword in query_lower for keyword in graph_keywords):
        return True

    multi_word_phrases = {
        "bar chart",
        "line graph",
        "scatter plot",
        "box plot",
        "data visualization",
        "figure showing",
        "chart showing",
    }
    for phrase in multi_word_phrases:
        if phrase in query_lower:
            return True

    return False


def chatbot_response_generator(user_query, page_name="Model Performance Analysis"):
    """Generator that simulates streaming response"""
    initialize_assistant()

    # Modify query if graph-related
    if is_graph_related(user_query):
        modified_query = f"In page '{page_name}', {user_query}"
    else:
        modified_query = user_query

    # Get full response from assistant
    full_response = process_user_query(modified_query)

    # Simulate streaming response
    words = full_response.split()
    current_chunk = ""
    for word in words:
        current_chunk += word + " "
        yield current_chunk.strip()
        time.sleep(0.05)


# --- UI Components ---
@dataclass
class Message:
    """Class for keeping track of a chat message."""

    role: Literal["user", "assistant"]
    content: str


def init_chatbot_state():
    if "show_chat" not in st.session_state:
        st.session_state.show_chat = False
    if "messages" not in st.session_state:
        st.session_state.messages = []


def load_chat_css():
    chat_css = """
    <style>
    .chat-container {
        max-width: 600px;
        margin: 0 auto;
        padding: 20px;
        background: #202020;
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(0,0,0,0.1);
    }
    .chat-row {
        display: flex;
        margin: 5px;
        width: 100%;
        align-items: flex-end;
    }
    .row-reverse {
        flex-direction: row-reverse;
    }
    .chat-bubble {
        font-family: "Source Sans Pro", sans-serif;
        border: 1px solid transparent;
        padding: 8px 15px;
        margin: 5px 7px;
        max-width: 70%;
        border-radius: 20px;
    }
    .assistant-bubble {
        color: black;
        background-color: #eeeeee;
        margin-right: 25%;
    }
    .user-bubble {
        color: white;
        background-color: #1F8AFF;
        margin-left: 25%;
    }
    .chat-icon {
        width: 28px !important;
        height: 28px !important;
        padding: 5px;
        margin-top: 5px !important;
        flex-shrink: 0;
    }
    .user-icon {
        color: rgb(31, 138, 255) !important;
    }
    .assistant-icon {
        color: rgb(64, 64, 64);
    }
    </style>
    """
    st.markdown(chat_css, unsafe_allow_html=True)


def get_chat_icon(role: str) -> str:
    if role == "user":
        return """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><path fill="currentColor" d="M12 4a3.5 3.5 0 1 0 0 7a3.5 3.5 0 0 0 0-7M6.5 7.5a5.5 5.5 0 1 1 11 0a5.5 5.5 0 0 1-11 0M3 19a5 5 0 0 1 5-5h8a5 5 0 0 1 5 5v3H3zm5-3a3 3 0 0 0-3 3v1h14v-1a3 3 0 0 0-3-3z"/></svg>"""
    else:
        return """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><path fill="currentColor" d="M18.5 10.255q0 .067-.003.133A1.54 1.54 0 0 0 17.473 10q-.243 0-.473.074V5.75a.75.75 0 0 0-.75-.75h-8.5a.75.75 0 0 0-.75.75v4.505c0 .414.336.75.75.75h8.276l-.01.025l-.003.012l-.45 1.384l-.01.026l-.019.053H7.75a2.25 2.25 0 0 1-2.25-2.25V5.75A2.25 2.25 0 0 1 7.75 3.5h3.5v-.75a.75.75 0 0 1 .649-.743L12 2a.75.75 0 0 1 .743.649l.007.101l-.001.75h3.5a2.25 2.25 0 0 1 2.25 2.25zm-5.457 3.781l.112-.036H6.254a2.25 2.25 0 0 0-2.25 2.25v.907a3.75 3.75 0 0 0 1.305 2.844c1.563 1.343 3.802 2 6.691 2c2.076 0 3.817-.339 5.213-1.028a1.55 1.55 0 0 1-1.169-1.003l-.004-.012l-.03-.093c-1.086.422-2.42.636-4.01.636c-2.559 0-4.455-.556-5.713-1.638a2.25 2.25 0 0 1-.783-1.706v-.907a.75.75 0 0 1 .75-.75H12v-.003a1.54 1.54 0 0 1 1.031-1.456zM10.999 7.75a1.25 1.25 0 1 0-2.499 0a1.25 1.25 0 0 0 2.499 0m3.243-1.25a1.25 1.25 0 1 1 0 2.499a1.25 1.25 0 0 1 0-2.499m1.847 10.912a2.83 2.83 0 0 0-1.348-.955l-1.377-.448a.544.544 0 0 1 0-1.025l1.377-.448a2.84 2.84 0 0 0 1.76-1.762l.01-.034l.449-1.377a.544.544 0 0 1 1.026 0l.448 1.377a2.84 2.84 0 0 0 1.798 1.796l1.378.448l.027.007a.544.544 0 0 1 0 1.025l-1.378.448a2.84 2.84 0 0 0-1.798 1.796l-.447 1.377a.55.55 0 0 1-.2.263a.544.544 0 0 1-.827-.263l-.448-1.377a2.8 2.8 0 0 0-.45-.848m7.694 3.801l-.765-.248a1.58 1.58 0 0 1-.999-.998l-.249-.765a.302.302 0 0 0-.57 0l-.249.764a1.58 1.58 0 0 1-.983.999l-.766.248a.302.302 0 0 0 0 .57l.766.249a1.58 1.58 0 0 1 .999 1.002l.248.764a.303.303 0 0 0 .57 0l.25-.764a1.58 1.58 0 0 1 .998-.999l.766-.248a.302.302 0 0 0 0-.57z"/></svg>"""


def create_message_div(msg: Message) -> str:
    icon = get_chat_icon(msg.role)
    chat_icon_class = (
        f"chat-icon {'user-icon' if msg.role == 'user' else 'assistant-icon'}"
    )
    return f"""
    <div class="chat-row {'row-reverse' if msg.role == 'user' else ''}">
        <div class="chat-icon-container">
            <div class="{chat_icon_class}">{icon}</div>
        </div>
        <div class="chat-bubble {'user-bubble' if msg.role == 'user' else 'assistant-bubble'}">
            ​{msg.content}
        </div>
    </div>
    """


# --- Floating Chat Interface ---
float_init()


def show_chatbot_ui(page_name="Model Performance Analysis"):
    init_chatbot_state()

    # Floating toggle button
    button_container = st.container()
    with button_container:
        btn_label = "⬇ Minimize" if st.session_state.show_chat else "💬 Ask AI"
        btn_type = "primary" if st.session_state.show_chat else "secondary"
        if st.button(btn_label, key="chat_toggle", type=btn_type):
            st.session_state.show_chat = not st.session_state.show_chat
            st.rerun()

    # Button styling
    button_css = float_css_helper(
        width="2.5rem", height="2.5rem", right="6rem", bottom="4rem", transition=0.1
    )
    button_container.float(button_css)

    # Chat container positioning
    chat_bottom = "7rem" if st.session_state.show_chat else "-500px"
    chat_transition = "all 0.5s cubic-bezier(0, 1, 0.5, 1)"

    # Chat container
    chat_container = st.container()
    chat_css = float_css_helper(
        width="min(75vw, 400px)",
        right="2rem",
        bottom=chat_bottom,
        css=f"padding: 1rem; {chat_transition}; background: white; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.2); max-height: 60vh; overflow-y: auto;",
        shadow=12,
    )
    chat_container.float(chat_css)

    if st.session_state.show_chat:
        with chat_container:
            load_chat_css()
            st.header("💬 AI Chatbot")
            messages_container = st.container(height=400)

            with st.container():
                if user_query := st.chat_input("Enter your question:"):
                    user_message = Message(role="user", content=user_query)
                    st.session_state.messages.append(user_message)

                    with messages_container:
                        for msg in st.session_state.messages:
                            st.markdown(create_message_div(msg), unsafe_allow_html=True)
                        streaming_placeholder = st.empty()

                    # Process query
                    with messages_container:
                        with st.spinner("Thinking..."):
                            response_generator = chatbot_response_generator(
                                user_query, page_name
                            )
                            full_response = ""
                            try:
                                first_chunk = next(response_generator)
                                full_response = first_chunk
                                streaming_div = f"""
                                <div class="chat-row">
                                    <div class="chat-icon-container">
                                        <div class="chat-icon assistant-icon">{get_chat_icon("assistant")}</div>
                                    </div>
                                    <div class="chat-bubble assistant-bubble">
                                        {full_response}
                                    </div>
                                </div>
                                """
                                streaming_placeholder.markdown(
                                    streaming_div, unsafe_allow_html=True
                                )
                            except StopIteration:
                                pass

                    # Stream response
                    for chunk in response_generator:
                        full_response = chunk
                        streaming_div = f"""
                        <div class="chat-row">
                            <div class="chat-icon-container">
                                <div class="chat-icon assistant-icon">{get_chat_icon("assistant")}</div>
                            </div>
                            <div class="chat-bubble assistant-bubble">
                                {full_response}
                            </div>
                        </div>
                        """
                        streaming_placeholder.markdown(
                            streaming_div, unsafe_allow_html=True
                        )

                    # Add final message
                    final_message = Message(role="assistant", content=full_response)
                    st.session_state.messages.append(final_message)
                    st.rerun()

            # Display message history
            if not st.session_state.messages:
                with messages_container:
                    st.markdown("", unsafe_allow_html=True)
            else:
                with messages_container:
                    for msg in st.session_state.messages:
                        st.markdown(create_message_div(msg), unsafe_allow_html=True)


if __name__ == "__main__":
    show_chatbot_ui()


# working with files upload each query
