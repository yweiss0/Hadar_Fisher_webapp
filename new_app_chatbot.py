import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv
import time
from streamlit_float import *
from dataclasses import dataclass
from typing import Literal
import os
import PyPDF2
import numpy as np
import asyncio
import nest_asyncio
import networkx as nx
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# Import tiktoken for token counting (install with: pip install tiktoken)
import tiktoken

# Load environment variables
load_dotenv()

# Initialize floating feature
float_init()

# Configuration
nest_asyncio.apply()

# ---------------- Global Variable to Accumulate Query Costs ----------------
QUERY_COSTS = []

# ---------------- Wrapped Functions with Cost Logging ----------------

DEFAULT_RAG_DIR = "new_lightrag_working_dir"
WORKING_DIR = os.environ.get("RAG_DIR", f"{DEFAULT_RAG_DIR}")
# print(f"WORKING_DIR: {WORKING_DIR}")
LLM_MODEL = "gpt-4o-mini"
# print(f"LLM_MODEL: {LLM_MODEL}")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
# print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
EMBEDDING_MAX_TOKEN_SIZE = int(os.environ.get("EMBEDDING_MAX_TOKEN_SIZE", 8192))
# print(f"EMBEDDING_MAX_TOKEN_SIZE: {EMBEDDING_MAX_TOKEN_SIZE}")
BASE_URL = "https://api.openai.com/v1"
# print(f"BASE_URL: {BASE_URL}")

# use in streamlit prod
API_KEY = st.secrets["OPENAI_API_KEY"]
# use in dev
# API_KEY = os.environ.get("OPENAI_API_KEY", "xxxxxxxx")
# print(f"API_KEY: {API_KEY}")

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


def get_token_count(text: str, model: str) -> int:
    """
    Count tokens for a given text using tiktoken.
    Falls back to a default encoding if the model isn’t recognized.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)


def estimate_llm_cost(
    query_tokens: int, completion_tokens: int, cached: bool = False
) -> float:
    """
    Estimate the cost of an LLM API call given the token counts.
    Prices:
      - Non-cached input: $0.150 per 1M tokens
      - Cached input:     $0.075 per 1M tokens
      - Output:           $0.600 per 1M tokens
    """
    if cached:
        input_rate = 0.075 / 1e6
    else:
        input_rate = 0.150 / 1e6
    output_rate = 0.600 / 1e6
    return query_tokens * input_rate + completion_tokens * output_rate


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        model=LLM_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=BASE_URL,
        api_key=API_KEY,
        **kwargs,
    )


async def llm_model_func_with_cost(
    prompt,
    system_prompt=None,
    history_messages=[],
    keyword_extraction=False,
    cached: bool = False,
    **kwargs,
) -> str:
    full_prompt = ""
    if system_prompt:
        full_prompt += system_prompt + "\n"
    full_prompt += prompt + "\n"
    for message in history_messages:
        if isinstance(message, dict) and "content" in message:
            full_prompt += message["content"] + "\n"
    query_tokens = get_token_count(full_prompt, LLM_MODEL)

    result = await llm_model_func(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        **kwargs,
    )
    completion_tokens = get_token_count(result, LLM_MODEL)
    total_tokens = query_tokens + completion_tokens
    estimated_cost = estimate_llm_cost(query_tokens, completion_tokens, cached=cached)
    # print(
    #     f"[LLM] Query tokens: {query_tokens}, Completion tokens: {completion_tokens}, Total tokens: {total_tokens}, Estimated cost: ${estimated_cost:.8f}"
    # )

    global QUERY_COSTS
    QUERY_COSTS.append(estimated_cost)
    return result


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts=texts,
        model=EMBEDDING_MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
    )


async def embedding_func_with_cost(texts: list[str]) -> np.ndarray:
    total_tokens = 0
    for text in texts:
        tokens = get_token_count(text, EMBEDDING_MODEL)
        total_tokens += tokens
        # print(f"[Embedding] Text snippet: {text[:30]}... Token count: {tokens}")
    result = await openai_embed(
        texts=texts,
        model=EMBEDDING_MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
    )
    # print(f"[Embedding] Total tokens: {total_tokens}")
    return result


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(texts=test_text)
    embedding_dim = embedding.shape[1]
    # print(f"{embedding_dim=}")
    return embedding_dim


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
    }
    for phrase in multi_word_phrases:
        if phrase in query_lower:
            return True

    if ("explain" in words or "what" in words) and any(
        keyword in query_lower for keyword in graph_keywords
    ):
        return True

    return False


rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func_with_cost,
    embedding_func=EmbeddingFunc(
        embedding_dim=asyncio.run(get_embedding_dim()),
        max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
        func=embedding_func_with_cost,
    ),
)


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
        font-family: "Source Sans Pro", sans-serif, "Segoe UI", "Roboto", sans-serif;
        border: 1px solid transparent;
        padding: 8px 15px;
        margin: 5px 7px;
        max-width: 70%;
        position: relative;
        border-radius: 20px;
    }
    .assistant-bubble {
        background-color: #eee;
        margin-right: 25%;
        position: relative;
    }
    .assistant-bubble:before {
        content: "";
        position: absolute;
        z-index: 0;
        bottom: 0;
        left: -7px;
        height: 20px;
        width: 20px;
        background: #eee;
        border-bottom-right-radius: 15px;
    }
    .assistant-bubble:after {
        content: "";
        position: absolute;
        z-index: 1;
        bottom: 0;
        left: -10px;
        width: 10px;
        height: 20px;
        background: white;
        border-bottom-right-radius: 10px;
    }
    .user-bubble {
        color: white;
        background-color: #1F8AFF;
        margin-left: 25%;
        position: relative;
        align-items: flex-end;
    }
    .user-bubble:before {
        content: "";
        position: absolute;
        z-index: 0;
        bottom: 0;
        right: -7px;
        height: 20px;
        width: 20px;
        background: #1F8AFF;
        border-bottom-left-radius: 15px;
        align-items: flex-end;
    }
    .user-bubble:after {
        content: "";
        position: absolute;
        z-index: 1;
        bottom: 0;
        right: -11px;
        width: 10px;
        height: 20px;
        background: white;
        border-bottom-left-radius: 10px;
        align-items: flex-end;
    }
    .chat-icon {
        width: 28px !important;
        height: 28px !important;
        min-width: 28px !important;
        min-height: 28px !important;
        padding: 5px;
        margin-top: 5px !important;
        flex-shrink: 0;
    }
    .user-icon {
        color: rgb(31, 138, 255) !important;
    }
    .assistant-icon {
        color: rgb(64, 64, 64);
        padding-left: 2px;
    }
    </style>
    """
    st.markdown(chat_css, unsafe_allow_html=True)


def load_scroll_css():
    scroll_css = """
    <style>
    #messages-container {
        max-height: 300px;
        overflow-y: auto;
        padding-right: 10px;
    }
    </style>
    """
    st.markdown(scroll_css, unsafe_allow_html=True)


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


def chatbot_response_generator(user_query, page_name="Model Performance Analysis"):
    """
    Yields chatbot responses in chunks for streaming effect.
    If the query is graph-related, prepends the page name to the query.
    """
    if is_graph_related(user_query):
        modified_query = f"In page '{page_name}', {user_query}, Limit your response to 70 words or fewer."
    else:
        modified_query = f"{user_query}, Limit your response to 70 words or fewer."

    hybrid_answer = rag.query(
        modified_query,
        param=QueryParam(mode="hybrid"),
    )
    # print(f"Query sent to RAG: {modified_query}")
    # print(f"RAG response: {hybrid_answer}")

    words = hybrid_answer.split()
    current_chunk = ""
    for word in words:
        current_chunk += word + " "
        yield current_chunk.strip()
        time.sleep(0.1)


def show_chatbot_ui(page_name="Model Performance Analysis"):
    init_chatbot_state()
    button_container = st.container()
    with button_container:
        btn_label = "⬇ Minimize" if st.session_state.show_chat else "💬 Ask AI"
        btn_type = "primary" if st.session_state.show_chat else "secondary"
        if st.button(btn_label, key="chat_toggle", type=btn_type):
            st.session_state.show_chat = not st.session_state.show_chat
            st.rerun()

    button_css = float_css_helper(
        width="2.5rem", height="2.5rem", right="6rem", bottom="4rem", transition=0.1
    )
    button_container.float(button_css)

    chat_bottom = "7rem" if st.session_state.show_chat else "-500px"
    chat_transition = "all 0.5s cubic-bezier(0, 1, 0.5, 1)"

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

                    # Show spinner while waiting for the LLM response
                    with messages_container:
                        with st.spinner("Thinking..."):
                            response_generator = chatbot_response_generator(
                                user_query, page_name
                            )
                            full_response = ""
                            # Get the first chunk to exit the spinner context
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

                    # Continue streaming the rest of the response
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

                    final_message = Message(role="assistant", content=full_response)
                    st.session_state.messages.append(final_message)
                    st.rerun()

            if not st.session_state.messages:
                with messages_container:
                    st.markdown("", unsafe_allow_html=True)
            else:
                with messages_container:
                    for msg in st.session_state.messages:
                        st.markdown(create_message_div(msg), unsafe_allow_html=True)
