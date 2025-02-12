import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI
import time
from streamlit_float import *
from dataclasses import dataclass
from typing import Literal



# Load environment variables
load_dotenv()

# Initialize floating feature
float_init()
#get the api key from the streamlit cloud secrets
API_KEY = st.secrets["OPENROUTER_API_KEY"]

# # Initialize session states
# if "show_chat" not in st.session_state:
#     st.session_state.show_chat = False

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# Configuration
PDF_PATH = "docs\draft_analysis.pdf"
# PDF_PATH = r"C:\Projects\personal_projects2\Hadar_Fisher_Website\docs\draft_analysis.pdf"
INDEX_NAME = "hadar_faiss_index"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "microsoft/phi-3-medium-128k-instruct:free"

# # Load CSV file
# @st.cache_data
# def load_data():
#     return pd.read_csv(r"C:\Projects\personal_projects2\Hadar_Fisher_Website\LDA6_modelfit_rf_nervous.csv")

# Chat bot logic
# Create Message class for chat tracking
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

# Load CSS function
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
    width: 28px !important; /* Set a fixed width */
    height: 28px !important; /* Set a fixed height */
    min-width: 28px !important; /* Prevent shrinking */
    min-height: 28px !important; /* Prevent shrinking */
    padding: 5px;
    margin-top: 5px !important;
    flex-shrink: 0; /* Ensures it doesn't shrink when messages stream */
    
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
    """
    Ensures that the messages container has a fixed height
    and scrolls automatically to the bottom.
    """
    scroll_css = """
    <style>
    #messages-container {
        max-height: 300px; /* or whatever fits your design */
        overflow-y: auto;
        padding-right: 10px; /* to avoid overlapping the scrollbar */
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
    chat_icon_class = f"chat-icon {'user-icon' if msg.role == 'user' else 'assistant-icon'}"
    return f"""
    <div class="chat-row {'row-reverse' if msg.role == 'user' else ''}">
        <div class="chat-icon-container">
            <div class="{chat_icon_class}">{icon}</div>
        </div>
        <div class="chat-bubble {'user-bubble' if msg.role == 'user' else 'assistant-bubble'}">
            &#8203;{msg.content}
        </div>
    </div>
    """

# Function to load FAISS vector store
def load_vectorstore():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.load_local(
        INDEX_NAME,
        embeddings,
        openai_api_key=OPENAI_API_KEY,
        allow_dangerous_deserialization=True
    )

# Function to set up LLM
def setup_llm():
    return ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=API_KEY,
        model_name=LLM_MODEL,
        max_tokens=1024,
        temperature=0.7
    )

# Function to handle chatbot response streaming
def chatbot_response_generator(user_query):
    """Yields chatbot responses in chunks for streaming effect."""
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever()
    llm = setup_llm()
    prompt = ChatPromptTemplate.from_template(
        """Answer the following question based only on the provided context:
        
        <context>
        {context}
        </context>
        
        Question: {input}.

        the context provided earlier seems to be related to a statistical analysis or research study topic.if It does not provide any direct relevance to the question, please ignore it and answer sorry i can't help with that."""
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": user_query})
    answer = response["answer"]
    
    for i in range(1, len(answer) + 1):
        yield answer[:i]
        time.sleep(0.02)  # Small delay for streaming effect

# end of chat bot logic

def show_chatbot_ui():
    # Ensure session state is set up:
    init_chatbot_state()
    # Chat Interface
    button_container = st.container()
    with button_container:
        btn_label = "⬇ Minimize" if st.session_state.show_chat else "💬 Ask AI"
        btn_type = "primary" if st.session_state.show_chat else "secondary"
        if st.button(btn_label, key="chat_toggle", type=btn_type):
            st.session_state.show_chat = not st.session_state.show_chat
            st.rerun()

    # Position the toggle button
    button_css = float_css_helper(
        width="2.5rem",
        height="2.5rem",
        right="6rem",
        bottom="2rem",
        transition=0.1
    )
    button_container.float(button_css)

    # Chat container positioning
    chat_bottom = "7rem" if st.session_state.show_chat else "-500px"
    chat_transition = "all 0.5s cubic-bezier(0, 1, 0.5, 1)"

    chat_container = st.container()
    chat_css = float_css_helper(
        width="400px",
        right="2rem",
        bottom=chat_bottom,
        css=f"padding: 1rem; {chat_transition}; background: white; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.2);",
        shadow=12
    )
    chat_container.float(chat_css)

    # Chat interface implementation
    if st.session_state.show_chat:
        with chat_container:
            # Load custom CSS
            load_chat_css()
            
            # Chat header
            st.header("💬 AI Chatbot")
            
            # Messages container
            messages_container = st.container(height=400)
            
            # Chat input
            with st.container():
                if user_query := st.chat_input("Enter your question:"):
                    # Add user message
                    user_message = Message(role="user", content=user_query)
                    st.session_state.messages.append(user_message)
                    
                    # Generate and stream response
                    response_generator = chatbot_response_generator(user_query)
                    full_response = ""
                    
                    # Create a placeholder for all messages including the streaming one
                    with messages_container:
                        for msg in st.session_state.messages:
                            st.markdown(create_message_div(msg), unsafe_allow_html=True)
                        
                        # Create a placeholder for the streaming message
                        streaming_placeholder = st.empty()
                    
                    # Stream the response with custom styling
                    for chunk in response_generator:
                        full_response = chunk
                        # Update only the streaming message
                        streaming_div = f"""
        <div class="chat-row">
            <div class="chat-icon-container">
                <div class="chat-icon assistant-icon">{get_chat_icon("assistant")}</div>
            </div>
            <div class="chat-bubble assistant-bubble">
                &#8203;{chunk}
            </div>
        </div>
        """
                        streaming_placeholder.markdown(streaming_div, unsafe_allow_html=True)
                    
                    # Add final response to session state
                    final_message = Message(role="assistant", content=full_response)
                    st.session_state.messages.append(final_message)
                    st.rerun()
            
            # Display existing messages when no new input
            if not st.session_state.messages:
                st.markdown("", unsafe_allow_html=True)
            else:
                with messages_container:
                    for msg in st.session_state.messages:
                        st.markdown(create_message_div(msg), unsafe_allow_html=True)

# WORKING WITH CUSTOM CSS and with option to import in different






