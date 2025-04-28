import streamlit as st
import os
import time
import traceback
from openai import OpenAI
from typing import List, Dict, Any
from dataclasses import dataclass
from typing import Literal
from streamlit_float import *
import re
import markdown
import numpy as np
import faiss
import pickle
import PyPDF2
import json
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
print("Loading environment variables from .env file...")

# --- Configuration ---
ASSISTANT_NAME = "PDF_QA_Assistant"
ASSISTANT_MODEL = "gpt-4.1-mini"
ASSISTANT_INSTRUCTIONS = (
    "You are a helpful on-site guide for research about emotion modeling. Your primary function is to answer questions "
    "about the PDF documents in the research database, including papers, graphs, visualizations, and methodological explanations. "
    "Only use information from these documents - do not use external knowledge. Keep your responses concise, around 70 words. "
    "If you cannot find information to answer a question in the provided context, clearly state: 'I can't help with that.' "
    "When explaining visualizations or graphs, be specific about what the data shows. Cite the document source when possible."
)

HELP_RESPONSE_TEXT = """💬 Hi, I'm your On-Site Guide: The Emotion Modeling Chatbot\n\n
I'm here to help you navigate the paper and the website's visualizations. I'm a helpful assistant that can answer many of the questions you might have while exploring the research.\n\n
---\n\n
📄 **I can Answer Questions About the Paper like**\n
* Responds to questions about the study's goals, methods, results, and conclusions\n
* Can explain key concepts from the paper in simpler terms\n
* Help clarify terminology, variables, and analytic approaches\n\n
📊 **I can Explain Website Figures, for example**\n
* Offer descriptions of the different interactive visualizations\n
* Help you understand what each figure shows (e.g., model performance, feature importance)\n
* Guide you in interpreting metrics like R², SHAP values, and model comparisons"""

POLLING_INTERVAL_S = 3
PDF_FILES = [
    "docs/new_draft_bot_25-03-25.pdf",
    "docs/graphs2.pdf",
    "docs/LIWC_vars.pdf",
    "docs/website_description.pdf",
    "docs/QnA_chatbot.pdf",
    "docs/onsiteguide.pdf",
]

# Vector DB configuration
VECTOR_DB_PATH = "docs/vector_store.faiss"
VECTOR_DB_METADATA_PATH = "docs/vector_store_metadata.pkl"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 5

# Initialize OpenAI client
API_KEY = None
try:
    # First try to get from Streamlit secrets
    API_KEY = st.secrets.get("OPENAI_API_KEY")
    if API_KEY:
        print("Using API Key from Streamlit secrets.")
    else:
        # Then try environment variable
        API_KEY = os.getenv("OPENAI_API_KEY")
        if API_KEY:
            print(
                f"Using API Key from environment variable. Key starts with: {API_KEY[:5]}..."
            )
        else:
            print("API Key not found in environment variables.")
except Exception as e:
    print(f"Error accessing Streamlit secrets: {e}")
    # Try environment variable as fallback
    API_KEY = os.getenv("OPENAI_API_KEY")
    if API_KEY:
        print(
            f"Using API Key from environment variable after secrets error. Key starts with: {API_KEY[:5]}..."
        )

# --- Initialize OpenAI Client ---
client = None
if API_KEY:
    try:
        client = OpenAI(api_key=API_KEY)
        # Test the client with a simple call
        response = client.embeddings.create(
            input="Test connection", model=EMBEDDING_MODEL
        )
        print("OpenAI client initialized and tested successfully.")
    except Exception as e:
        error_message = f"Failed to initialize or test OpenAI client: {e}"
        print(error_message)
        st.error(error_message)
        # client remains None
else:
    error_message = "OpenAI API Key not found. Please set it in Streamlit secrets (secrets.toml) or as an environment variable OPENAI_API_KEY."
    print(error_message)
    st.error(error_message)


# --- RAG Vector Database Functions ---
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        text = ""
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """Split text into overlapping chunks."""
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        # If this is not the last chunk, try to end at a period or newline
        if end < text_length:
            # Look for a good breaking point (period followed by space, or newline)
            last_period = max(
                text.rfind(". ", start, end), text.rfind("\n", start, end)
            )
            if (
                last_period > start + 0.5 * chunk_size
            ):  # Only use if we've covered at least half the desired chunk
                end = last_period + 1  # Include the period

        chunks.append(text[start:end])
        start = end - overlap

    return chunks


def get_embeddings(texts: List[str]) -> np.ndarray:
    """Get embeddings for a list of texts using OpenAI API."""
    embeddings = []

    # Check if client is initialized
    if client is None:
        print("ERROR: OpenAI client is not initialized for getting embeddings.")
        # Return a dummy embedding of zeros
        return np.zeros((len(texts), 1536), dtype=np.float32)

    for text in tqdm(texts, desc="Generating embeddings"):
        if not text.strip():  # Skip empty texts
            continue
        try:
            response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Add a zero vector as a placeholder
            embeddings.append([0.0] * 1536)  # Standard OpenAI embedding dimension

    return np.array(embeddings, dtype=np.float32)


def build_vector_database() -> None:
    """Build a FAISS vector database from PDF files."""
    if os.path.exists(VECTOR_DB_PATH) and os.path.exists(VECTOR_DB_METADATA_PATH):
        print("Vector database already exists. Skipping build.")
        return

    print(
        "Vector database not found. Please run build_vector_db.py first to create it."
    )
    st.error(
        "Vector database not found. Please run build_vector_db.py first to create it."
    )
    return


def load_vector_database():
    """Load the FAISS vector database and metadata."""
    try:
        if not os.path.exists(VECTOR_DB_PATH) or not os.path.exists(
            VECTOR_DB_METADATA_PATH
        ):
            print(
                "Vector database files not found. Please run build_vector_db.py first."
            )
            return None, [], []

        print("Loading vector database from disk...")
        index = faiss.read_index(VECTOR_DB_PATH)
        with open(VECTOR_DB_METADATA_PATH, "rb") as f:
            metadata, chunks = pickle.load(f)
        print(
            f"Vector database loaded with {len(chunks)} chunks and {index.ntotal} vectors."
        )
        return index, metadata, chunks
    except Exception as e:
        print(f"Error loading vector database: {e}")
        return None, [], []


def search_vector_database(
    query: str, top_k: int = TOP_K_RESULTS
) -> List[Dict[str, Any]]:
    """Search the vector database for similar chunks."""
    # Load the database
    index, metadata, chunks = load_vector_database()
    if index is None:
        return []

    # Get query embedding
    print(f"Generating embedding for query: '{query[:50]}...'")
    try:
        query_embedding = get_embeddings([query])[0].reshape(1, -1)

        # Search
        print(f"Searching for top {top_k} matches...")
        distances, indices = index.search(query_embedding, top_k)

        # Return results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(metadata):
                results.append(
                    {
                        "content": chunks[idx],
                        "metadata": metadata[idx],
                        "score": float(distances[0][i]),
                    }
                )

        print(f"Found {len(results)} matching results.")
        return results
    except Exception as e:
        print(f"Error during vector search: {e}")
        return []


def initialize_vector_database():
    """Initialize the vector database if not already present."""
    if "vector_db_initialized" not in st.session_state:
        if not os.path.exists(VECTOR_DB_PATH) or not os.path.exists(
            VECTOR_DB_METADATA_PATH
        ):
            print("Vector database not found. Please run build_vector_db.py first.")
            st.warning(
                "Vector database not found. Please run build_vector_db.py first to create it."
            )
        else:
            print("Vector database found. Ready to use.")
        st.session_state.vector_db_initialized = True


# --- Core Chatbot Functions ---
def upload_files(file_paths: List[str]) -> List[str]:
    """Uploads multiple files to OpenAI and returns their IDs."""
    uploaded_file_ids = []
    # Check if client is properly initialized
    if client is None:
        print("OpenAI client is not initialized. Skipping file upload.")
        return uploaded_file_ids

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


def is_help_query(query: str) -> bool:
    """Detects if user is asking about chatbot capabilities"""
    help_phrases = {
        "what can you do",
        "what do you know",
        "what can i ask",
        "capabilities",
        "purpose",
        "function",
        "assist with",
        "what are you",
        "who are you",
        "explain yourself",
        "what questions can i ask",
        "how to use",
        "what do you do",
        "what can you help with",
        "waht can you do?",
    }

    query = query.lower().strip("?.,! ")
    return any(phrase in query for phrase in help_phrases) or query in {
        "what can you help with",
        "what can you do",
    }


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
        # Check if client is properly initialized
        if client is None:
            print(
                "OpenAI client is not initialized. Skipping assistant initialization."
            )
            st.session_state.assistant_initialized = False
            st.session_state.uploaded_file_ids = []
            return

        # Upload files
        st.session_state.uploaded_file_ids = upload_files(PDF_FILES)

        try:
            # Create assistant
            assistant = client.beta.assistants.create(
                name=ASSISTANT_NAME,
                instructions=ASSISTANT_INSTRUCTIONS,
                model=ASSISTANT_MODEL,
                tools=[{"type": "file_search"}],
            )
            st.session_state.assistant_id = assistant.id
            st.session_state.assistant_initialized = True
        except Exception as e:
            print(f"Error creating assistant: {e}")
            st.session_state.assistant_initialized = False


def process_user_query(question: str) -> str:
    """Process user query using RAG and OpenAI"""
    # Initialize vector database if needed
    initialize_vector_database()

    # Check if vector database exists before proceeding with RAG
    if not os.path.exists(VECTOR_DB_PATH) or not os.path.exists(
        VECTOR_DB_METADATA_PATH
    ):
        # Fallback to the original Assistants API if vector database is missing
        if client is not None:
            return process_user_query_via_assistant_api(question)
        else:
            return "Vector database not found and OpenAI client not initialized. Unable to process your request."

    # Search vector database for relevant contexts
    search_results = search_vector_database(question)

    # Format the results as context
    context = ""
    if search_results:
        context = "Here is information from the research documents:\n\n"
        for i, result in enumerate(search_results):
            content = result["content"]
            # Remove the source information from the context to prevent model confusion
            context += f"[Document {i+1}]:\n{content}\n\n"
    else:
        # No relevant context found
        return "I couldn't find relevant information to answer your question. Could you try rephrasing or asking something else about the research?"

    # Construct the prompt with retrieved context
    rag_prompt = f"""
The user asked: "{question}"

Here is relevant information from the research:

{context}

Based ONLY on the information provided above, answer the user's question concisely (about 70 words).
If the answer cannot be clearly found in the provided context, respond with "I can't help with that."
Be specific and precise, especially when explaining research methods, results, or visualizations.

IMPORTANT: DO NOT mention or reference any document numbers, file names, or sources in your answer. 
DO NOT include any text like "According to Document X" or "From the source file" or "As stated in PDF X" or similar phrases.
DO NOT use citations or references like (Document 1), (Source: X), etc.
Simply provide the answer directly as if you already knew it.
"""

    try:
        if client is None:
            return "OpenAI client is not initialized. Unable to process your request."

        # Use a direct completion API call for RAG
        response = client.chat.completions.create(
            model=ASSISTANT_MODEL,
            messages=[
                {"role": "system", "content": ASSISTANT_INSTRUCTIONS},
                {"role": "user", "content": rag_prompt},
            ],
        )
        answer = response.choices[0].message.content.strip()

        # Post-process the answer to remove any remaining source references
        answer = remove_source_references(answer)
        return answer
    except Exception as e:
        print(f"Error with RAG completion: {e}")
        # Fallback to the original Assistants API if RAG fails
        if client is not None:
            return process_user_query_via_assistant_api(question)
        else:
            return f"Error processing your request: {str(e)}"


def remove_source_references(text: str) -> str:
    """Remove any source references from the text."""
    # Pattern for document references like "(Document 1)" or "Document 2:"
    text = re.sub(r"\(Document\s+\d+[,\s]*\d*[,\s]*\d*\)", "", text)
    text = re.sub(r"Document\s+\d+[,\s]*\d*[,\s]*\d*:", "", text)

    # Pattern for source file references like "(Source: filename.pdf)" or "from filename.pdf"
    text = re.sub(r"\(Source:?\s*[^)]*\.pdf\)", "", text)
    text = re.sub(r"from\s+[^,\s]*\.pdf", "", text)

    # Pattern for general source references
    text = re.sub(r"\(Sources?:.*?\)", "", text)
    text = re.sub(r"According to the documents?[,:]", "", text)
    text = re.sub(r"Based on the (provided )?documents?[,:]", "", text)

    # Pattern for specific PDF references
    for pdf_file in PDF_FILES:
        basename = os.path.basename(pdf_file)
        text = text.replace(basename, "")
        text = text.replace(os.path.splitext(basename)[0], "")

    # Clean up any double spaces or unnecessary punctuation that might result
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s+:", ":", text)
    text = re.sub(r"\.+", ".", text)

    return text.strip()


def process_user_query_via_assistant_api(question: str) -> str:
    """Process user query using the original Assistants API (fallback method)"""
    if client is None:
        return "OpenAI client is not initialized. Unable to process your request via Assistant API."

    if not st.session_state.get("assistant_initialized", False):
        return "Assistant is not initialized. Unable to process your request."

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
        response = get_assistant_response_text(st.session_state.thread_id)
        # Also apply source reference removal to the assistant API response
        return remove_source_references(response)
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
    # Check for help query first
    if is_help_query(user_query):
        # Simulate thinking delay
        time.sleep(2)

        # Split help text into lines for streaming
        help_lines = HELP_RESPONSE_TEXT.split("\n")
        full_response = ""

        # Stream character by character with appropriate delays
        for i, line in enumerate(help_lines):
            if i > 0:  # Add newline before new lines except the first one
                full_response += "\n"

            # Add line character by character
            for char in line:
                full_response += char
                yield full_response.strip()
                time.sleep(0.01)  # Adjust timing for smooth appearance

            # Add slight delay between sections
            if line.startswith(("📄", "📊")):
                time.sleep(0.2)
        return

    # Initialize vector database
    initialize_vector_database()

    # Existing logic for normal queries
    initialize_assistant()

    if is_graph_related(user_query):
        modified_query = f"In page '{page_name}', {user_query}"
    else:
        modified_query = user_query

    full_response = process_user_query(modified_query)

    # Apply source reference removal one final time before streaming
    full_response = remove_source_references(full_response)

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
        st.session_state.show_chat = True
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

    # Convert markdown to HTML if this is an assistant message
    content = markdown.markdown(msg.content) if msg.role == "assistant" else msg.content

    return f"""
    <div class="chat-row {'row-reverse' if msg.role == 'user' else ''}">
        <div class="chat-icon-container">
            <div class="{chat_icon_class}">{icon}</div>
        </div>
        <div class="chat-bubble {'user-bubble' if msg.role == 'user' else 'assistant-bubble'}">
            {content}
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
                            # Add artificial delay for help responses
                            if is_help_query(user_query):
                                time.sleep(2)

                            response_generator = chatbot_response_generator(
                                user_query, page_name
                            )
                            full_response = ""
                            try:
                                first_chunk = next(response_generator)
                                full_response = first_chunk
                                streaming_div = """
                                <div class="chat-row">
                                    <div class="chat-icon-container">
                                        <div class="chat-icon assistant-icon">{get_chat_icon("assistant")}</div>
                                    </div>
                                    <div class="chat-bubble assistant-bubble">
                                        {full_response.replace("\n", "<br>")}
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

                        # Convert markdown to HTML for display
                        html_response = markdown.markdown(full_response)

                        streaming_div = f"""
                        <div class="chat-row">
                            <div class="chat-icon-container">
                                <div class="chat-icon assistant-icon">{get_chat_icon("assistant")}</div>
                            </div>
                            <div class="chat-bubble assistant-bubble">
                                {html_response}
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
