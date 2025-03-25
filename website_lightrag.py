import os
import PyPDF2
import numpy as np
import asyncio
import nest_asyncio
import networkx as nx
import time
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from pyvis.network import Network
from transformers import AutoModel, AutoTokenizer

# Import tiktoken for token counting (install with: pip install tiktoken)
import tiktoken

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

# ---------------- Global Variable to Accumulate Query Costs ----------------
QUERY_COSTS = []

# ---------------- Helper Functions for Token Counting and Cost Estimation ----------------


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
        input_rate = 0.075 / 1e6  # $0.075 per token
    else:
        input_rate = 0.150 / 1e6  # $0.150 per token
    output_rate = 0.600 / 1e6  # $0.600 per token

    return query_tokens * input_rate + completion_tokens * output_rate


# ---------------- Wrapped Functions with Cost Logging ----------------

DEFAULT_RAG_DIR = "25-03-25-lightrag_data"
WORKING_DIR = os.environ.get("RAG_DIR", f"{DEFAULT_RAG_DIR}")
print(f"WORKING_DIR: {WORKING_DIR}")
LLM_MODEL = "gpt-4o-mini"
print(f"LLM_MODEL: {LLM_MODEL}")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
EMBEDDING_MAX_TOKEN_SIZE = int(os.environ.get("EMBEDDING_MAX_TOKEN_SIZE", 8192))
print(f"EMBEDDING_MAX_TOKEN_SIZE: {EMBEDDING_MAX_TOKEN_SIZE}")
BASE_URL = "https://api.openai.com/v1"
print(f"BASE_URL: {BASE_URL}")
API_KEY = os.environ.get("OPENAI_API_KEY", "xxxxxxxx")
print(f"API_KEY: {API_KEY}")

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


# Original LLM model function
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


# Wrapped LLM model function with token and cost logging
async def llm_model_func_with_cost(
    prompt,
    system_prompt=None,
    history_messages=[],
    keyword_extraction=False,
    cached: bool = False,
    **kwargs,
) -> str:
    # Build the full prompt text from all components for token counting
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
    print(
        f"[LLM] Query tokens: {query_tokens}, Completion tokens: {completion_tokens}, Total tokens: {total_tokens}, Estimated cost: ${estimated_cost:.8f}"
    )

    # Append this cost to the global QUERY_COSTS list
    global QUERY_COSTS
    QUERY_COSTS.append(estimated_cost)

    return result


# Original asynchronous embedding function
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts=texts,
        model=EMBEDDING_MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
    )


# Wrapped embedding function with token logging (add cost logic if needed)
async def embedding_func_with_cost(texts: list[str]) -> np.ndarray:
    total_tokens = 0
    for text in texts:
        tokens = get_token_count(text, EMBEDDING_MODEL)
        total_tokens += tokens
        print(f"[Embedding] Text snippet: {text[:30]}... Token count: {tokens}")
    result = await openai_embed(
        texts=texts,
        model=EMBEDDING_MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
    )
    print(f"[Embedding] Total tokens: {total_tokens}")
    return result


# ---------------- Obtain Embedding Dimension ----------------


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(texts=test_text)
    embedding_dim = embedding.shape[1]
    print(f"{embedding_dim=}")
    return embedding_dim


# ---------------- Initialize RAG Instance ----------------

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func_with_cost,  # Use the wrapped function to log cost info
    embedding_func=EmbeddingFunc(
        embedding_dim=asyncio.run(get_embedding_dim()),
        max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
        func=embedding_func_with_cost,  # Use the wrapped embedding function
    ),
)

# ---------------- Insert PDF Files ----------------

pdf_paths = [
    "docs/new_draft_bot_25-03-25.pdf",
    "docs/graphs2.pdf",
    "docs/LIWC_vars.pdf",
]

for pdf_path in pdf_paths:
    with open(pdf_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        rag.insert(text)

# ---------------- Performing Queries and Measuring Time ----------------

total_start = time.time()

# For each query, after it finishes we sum the costs from QUERY_COSTS and reset it.

# Naive query
# start = time.time()
# naive_answer = rag.query("explain me about the graph in page feature importance Heatmap (SHAP Values)",
#                          param=QueryParam(mode="naive"))
# naive_time = time.time() - start
# naive_cost = sum(QUERY_COSTS)
# QUERY_COSTS = []  # reset after query

# # Local query
# start = time.time()
# local_answer = rag.query("explain me about the graph in page feature importance Heatmap (SHAP Values)",
#                          param=QueryParam(mode="local"))
# local_time = time.time() - start
# local_cost = sum(QUERY_COSTS)
# QUERY_COSTS = []

# # Global query
# start = time.time()
# global_answer = rag.query("explain me about the graph in page feature importance Heatmap (SHAP Values)",
#                           param=QueryParam(mode="global"))
# global_time = time.time() - start
# global_cost = sum(QUERY_COSTS)
# QUERY_COSTS = []


# Mix query
# start = time.time()
# mix_answer = rag.query("explain me about the graph in page  feature importance Heatmap (SHAP Values)",
#                        param=QueryParam(mode="mix"))
# mix_time = time.time() - start
# mix_cost = sum(QUERY_COSTS)
# QUERY_COSTS = []

total_time = time.time() - total_start
# Hybrid query
start = time.time()
hybrid_answer = rag.query(
    "explain me what is Emotional Tone",
    param=QueryParam(mode="hybrid"),
)
hybrid_time = time.time() - start
hybrid_cost = sum(QUERY_COSTS)
QUERY_COSTS = []


# ---------------- Save Answers, Timings, and Costs to a Text File ----------------

output_file = os.path.join(WORKING_DIR, "answers_graph2.txt")
with open(output_file, "w", encoding="utf-8") as f:
    # f.write("naive:\n")
    # f.write(str(naive_answer) + "\n")
    # f.write("Time: {:.2f} seconds\n".format(naive_time))
    # f.write("Cost: ${:.8f}\n".format(naive_cost))
    # f.write("\n\n\n")

    # f.write("local:\n")
    # f.write(str(local_answer) + "\n")
    # f.write("Time: {:.2f} seconds\n".format(local_time))
    # f.write("Cost: ${:.8f}\n".format(local_cost))
    # f.write("\n\n\n")

    # f.write("global:\n")
    # f.write(str(global_answer) + "\n")
    # f.write("Time: {:.2f} seconds\n".format(global_time))
    # f.write("Cost: ${:.8f}\n".format(global_cost))
    # f.write("\n\n\n")

    # f.write("mix:\n")
    # f.write(str(mix_answer) + "\n")
    # f.write("Time: {:.2f} seconds\n".format(mix_time))
    # f.write("Cost: ${:.8f}\n".format(mix_cost))
    # f.write("\n\n\n")

    f.write("hybrid:\n")
    f.write(str(hybrid_answer) + "\n")
    f.write("Time: {:.2f} seconds\n".format(hybrid_time))
    f.write("Cost: ${:.8f}\n".format(hybrid_cost))
    f.write("\n\n\n")

    f.write("Total time for all queries: {:.2f} seconds\n".format(total_time))

print("All answers, timings, and costs saved to", output_file)

# ---------------- Generate and Display the Knowledge Graph ----------------

# # Load the GraphML file
# G = nx.read_graphml(os.path.join(WORKING_DIR, "graph_chunk_entity_relation.graphml"))

# # Create a Pyvis network
# net = Network(notebook=True)

# # Convert NetworkX graph to Pyvis network
# net.from_nx(G)

# # Save and display the network
# net.show(os.path.join(WORKING_DIR, "OPENAI_knowledge_graph.html"))
