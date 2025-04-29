import os
import time
import numpy as np
import faiss
import pickle
import PyPDF2
import json
from tqdm import tqdm
from openai import OpenAI
from typing import List, Dict, Any
import gc  # Garbage collector for memory management
from dotenv import load_dotenv
import sys
import argparse

# Load environment variables from .env file
load_dotenv()
print("Loading environment variables from .env file...")

print("Starting vector database build script...")

# Create argument parser
parser = argparse.ArgumentParser(
    description="Build or update vector database from PDF and TXT files"
)
parser.add_argument(
    "--mode",
    choices=["build", "update"],
    default="build",
    help='Mode: "build" for full rebuild, "update" to add new files only',
)
parser.add_argument("--files", nargs="+", help="Specific files to add (in update mode)")
parser.add_argument(
    "--force", action="store_true", help="Force rebuild even if database exists"
)
args = parser.parse_args()

# --- Configuration ---
# Default files list (will be processed in "build" mode)
DEFAULT_INPUT_FILES = [
    "docs/new_draft_bot_25-03-25.pdf",
    "docs/graphs2.pdf",
    "docs/LIWC_vars.pdf",
    "docs/website_description.pdf",
    "docs/QnA_chatbot.pdf",
    "docs/onsiteguide.pdf",
    "docs/article_writers.txt",
    # Add any default TXT files here
]

# Use provided files if in update mode with specific files
INPUT_FILES = (
    args.files if args.mode == "update" and args.files else DEFAULT_INPUT_FILES
)

# Vector DB configuration
VECTOR_DB_PATH = "docs/vector_store.faiss"
VECTOR_DB_METADATA_PATH = "docs/vector_store_metadata.pkl"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 800  # Reduced from 1000 to avoid memory issues
CHUNK_OVERLAP = 150  # Reduced from 200 to avoid memory issues

print(f"Running in {args.mode} mode")
if args.mode == "update":
    print(
        f"Files to process: {', '.join(INPUT_FILES) if INPUT_FILES else 'None provided'}"
    )
else:
    print(f"Will process {len(INPUT_FILES)} files.")

print(f"Vector DB will be saved to: {VECTOR_DB_PATH}")
print(f"Metadata will be saved to: {VECTOR_DB_METADATA_PATH}")
print(f"Using chunk size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}")

# Initialize OpenAI client
API_KEY = None
try:
    # Try to load from environment variable
    API_KEY = os.getenv("OPENAI_API_KEY")
    if API_KEY:
        print(
            f"Using API Key from environment variable. Key starts with: {API_KEY[:5]}..."
        )
    else:
        # If not in environment, ask user to input it
        API_KEY = input("Please enter your OpenAI API key: ")
        if not API_KEY:
            raise ValueError("No API key provided")
except Exception as e:
    print(f"Error with API key: {e}")
    exit(1)

# Initialize OpenAI Client
try:
    client = OpenAI(api_key=API_KEY)
    # Test the client with a simple call
    response = client.embeddings.create(input="Test connection", model=EMBEDDING_MODEL)
    print("OpenAI client initialized and tested successfully.")
except Exception as e:
    print(f"Failed to initialize or test OpenAI client: {e}")
    exit(1)


def sanitize_text(text: str) -> str:
    """
    Sanitize text to remove or replace problematic characters that could cause encoding issues.
    Specifically handles surrogate character issues with UTF-8 encoding.
    """
    if not text:
        return ""

    try:
        # Try to encode then decode as UTF-8 to catch and remove any invalid characters
        sanitized = text.encode("utf-8", "replace").decode("utf-8")

        # Remove control characters except newlines and tabs
        sanitized = "".join(
            char
            for char in sanitized
            if char == "\n" or char == "\t" or ord(char) >= 32
        )

        return sanitized
    except Exception as e:
        print(f"Error during text sanitization: {e}")
        # Extreme fallback: remove all non-ASCII characters
        return "".join(char for char in text if ord(char) < 128)


def extract_text_from_file(file_path: str) -> str:
    """Extract text from a file based on its extension."""
    print(f"Extracting text from {file_path}...")

    # Get file extension
    _, extension = os.path.splitext(file_path.lower())

    try:
        # Handle text files
        if extension == ".txt":
            return extract_text_from_txt(file_path)
        # Handle PDF files
        elif extension == ".pdf":
            return extract_text_from_pdf(file_path)
        else:
            print(
                f"Unsupported file type: {extension}. Currently supporting: .pdf, .txt"
            )
            return ""
    except Exception as e:
        print(f"  - ERROR processing file {file_path}: {e}")
        return ""


def extract_text_from_txt(txt_path: str) -> str:
    """Extract text from a plain text file."""
    print(f"  - Reading text file: {txt_path}")
    try:
        with open(txt_path, "r", encoding="utf-8") as file:
            text = file.read()

        # Sanitize text to handle encoding issues
        text = sanitize_text(text)
        print(f"  - Extraction complete. Extracted {len(text)} characters")
        return text
    except UnicodeDecodeError:
        # Try different encoding if UTF-8 fails
        try:
            with open(txt_path, "r", encoding="latin-1") as file:
                text = file.read()

            # Sanitize text to handle encoding issues
            text = sanitize_text(text)
            print(
                f"  - Extraction complete using latin-1 encoding. Extracted {len(text)} characters"
            )
            return text
        except Exception as e:
            print(f"  - ERROR extracting text from {txt_path}: {e}")
            return ""
    except Exception as e:
        print(f"  - ERROR extracting text from {txt_path}: {e}")
        return ""


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    print(f"  - Reading PDF file: {pdf_path}")
    try:
        text = ""
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            print(f"    - PDF has {total_pages} pages")

            for i, page in enumerate(pdf_reader.pages):
                if i % 5 == 0:  # Update every 5 pages
                    print(f"    - Processing page {i+1}/{total_pages}")
                try:
                    page_text = page.extract_text()
                    # Sanitize each page's text immediately after extraction
                    page_text = sanitize_text(page_text)
                    text += page_text + "\n"
                except Exception as e:
                    print(f"    - Error extracting text from page {i+1}: {e}")

            # Force garbage collection to free memory
            gc.collect()

        print(f"    - Extraction complete. Extracted {len(text)} characters")
        return text
    except Exception as e:
        print(f"    - ERROR extracting text from {pdf_path}: {e}")
        return ""


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """Split text into overlapping chunks with memory optimization."""
    if not text:
        return []

    print(
        f"Chunking text of {len(text)} characters (chunk size: {chunk_size}, overlap: {overlap})..."
    )
    chunks = []
    start = 0
    text_length = len(text)
    chunk_count = 0
    previous_start = -1  # To detect if we're stuck in a loop
    progress_interval = max(
        1, min(1000, text_length // chunk_size // 10)
    )  # Show progress about 10 times

    print(f"  - Estimated chunks: ~{(text_length - overlap) // (chunk_size - overlap)}")
    print("  - Starting chunking process...")

    while start < text_length:
        # Safety check - if we're not making progress, force advance
        if start == previous_start:
            print(
                f"  - WARNING: Detected loop at position {start}, advancing position..."
            )
            start += chunk_size // 2  # Force advance by half a chunk size
            if start >= text_length:
                break

        previous_start = start

        # Print progress updates
        if chunk_count % progress_interval == 0:
            progress_pct = min(100, int((start / text_length) * 100))
            print(
                f"  - Progress: {progress_pct}% - Chunk #{chunk_count} - Position {start}/{text_length}"
            )

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

        # Create chunk and add to list
        chunk = text[start:end]

        # Sanitize each chunk before adding it to the list
        chunk = sanitize_text(chunk)
        chunks.append(chunk)
        chunk_count += 1

        # Move to next position
        start = end - overlap

        # Safety check - ensure we make at least some minimum progress
        if start <= previous_start and end < text_length:
            start = previous_start + max(
                1, chunk_size // 10
            )  # Ensure at least 10% progress
            print(f"  - Adjusted position to ensure progress: {start}/{text_length}")

        # Limit maximum number of chunks as a safety measure
        if chunk_count > 10000:  # This is an arbitrary limit
            print(
                "  - WARNING: Reached maximum chunk count (10000). Breaking to avoid infinite loop."
            )
            break

        # Periodically force garbage collection to prevent memory issues
        if chunk_count % 100 == 0:
            gc.collect()

    print(f"  - Chunking complete! Created {len(chunks)} chunks")
    # Final garbage collection
    gc.collect()
    return chunks


def get_embeddings(texts: List[str]) -> np.ndarray:
    """Get embeddings for a list of texts using OpenAI API."""
    print(f"Generating embeddings for {len(texts)} text chunks...")
    embeddings = []

    # Process chunks in batches for progress updates
    batch_size = max(1, len(texts) // 10)  # Show progress ~10 times

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        print(
            f"  - Processing batch {i//batch_size + 1}/{(len(texts)+batch_size-1)//batch_size} ({i}/{len(texts)} chunks)"
        )

        for j, text in enumerate(batch):
            if not text.strip():  # Skip empty texts
                print(f"    - Skipping empty text at index {i+j}")
                continue
            try:
                # Sanitize text before generating embedding
                sanitized_text = sanitize_text(text)

                start_time = time.time()
                response = client.embeddings.create(
                    input=sanitized_text, model=EMBEDDING_MODEL
                )
                embeddings.append(response.data[0].embedding)
                elapsed = time.time() - start_time

                # Only print detailed progress for every 10th chunk to avoid too much output
                if (i + j) % 10 == 0 or j == len(batch) - 1:
                    print(
                        f"    - Generated embedding {i+j+1}/{len(texts)} ({elapsed:.2f}s)"
                    )

            except Exception as e:
                print(f"    - ERROR generating embedding for chunk {i+j}: {e}")
                print(f"    - Attempting with more aggressive sanitization...")
                try:
                    # Try with more aggressive ASCII-only sanitization
                    ascii_text = "".join(c for c in text if ord(c) < 128)
                    response = client.embeddings.create(
                        input=ascii_text, model=EMBEDDING_MODEL
                    )
                    embeddings.append(response.data[0].embedding)
                    print(
                        f"    - Successfully generated embedding with ASCII-only text"
                    )
                except Exception as retry_error:
                    print(f"    - Retry also failed: {retry_error}")
                    # Add a zero vector as a placeholder
                    embeddings.append(
                        [0.0] * 1536
                    )  # Standard OpenAI embedding dimension

            # Release memory periodically
            if (i + j) % 50 == 0:
                gc.collect()

    print(f"Embeddings generation complete. Created {len(embeddings)} embeddings.")
    return np.array(embeddings, dtype=np.float32)


def process_file_in_chunks(
    file_path: str, all_chunks: List[str], all_metadata: List[Dict]
):
    """Process a single file (PDF or text), adding chunks to the provided lists."""
    if not os.path.exists(file_path):
        print(f"WARNING: {file_path} not found, skipping...")
        return 0

    file_text = extract_text_from_file(file_path)
    if not file_text:
        print(f"No text extracted from {file_path}, skipping...")
        return 0

    chunks = chunk_text(file_text)

    # Free memory from the large text string
    del file_text
    gc.collect()

    chunk_count = len(chunks)
    chunk_base_idx = len(all_chunks)

    print(f"  - Adding {chunk_count} chunks to database...")
    for j, chunk in enumerate(chunks):
        # Ensure the chunk is sanitized before storing
        sanitized_chunk = sanitize_text(chunk)
        all_chunks.append(sanitized_chunk)
        all_metadata.append(
            {
                "source": file_path,
                "chunk_id": j,
                "content": sanitized_chunk[:100] + "...",  # Short preview
            }
        )

        # Print progress periodically
        if j % 50 == 0 and j > 0:
            print(f"  - Added {j}/{chunk_count} chunks")

    print(f"Added {chunk_count} chunks from {file_path}")
    return chunk_count


def load_existing_database():
    """Load the existing vector database."""
    try:
        if not os.path.exists(VECTOR_DB_PATH) or not os.path.exists(
            VECTOR_DB_METADATA_PATH
        ):
            print("Existing vector database not found.")
            return None, [], []

        print("Loading existing vector database...")
        index = faiss.read_index(VECTOR_DB_PATH)
        with open(VECTOR_DB_METADATA_PATH, "rb") as f:
            metadata, chunks = pickle.load(f)

        print(
            f"Loaded existing database with {len(chunks)} chunks and {index.ntotal} vectors."
        )
        return index, metadata, chunks
    except Exception as e:
        print(f"Error loading existing database: {e}")
        return None, [], []


def get_processed_files(metadata):
    """Extract the list of files that have already been processed."""
    processed_files = set()
    for item in metadata:
        if "source" in item:
            processed_files.add(item["source"])
    return processed_files


def update_vector_database(new_files) -> None:
    """Update the vector database by adding new files."""
    print("\n--- Updating Vector Database ---\n")

    # Load existing database
    index, metadata, chunks = load_existing_database()
    if index is None:
        print("Cannot update database as it doesn't exist. Run in 'build' mode first.")
        return

    # Get list of already processed files
    processed_files = get_processed_files(metadata)
    print(f"Found {len(processed_files)} already processed files.")

    # Filter out already processed files
    files_to_process = []
    for file_path in new_files:
        if file_path in processed_files:
            print(f"Skipping {file_path} as it's already in the database.")
        else:
            files_to_process.append(file_path)

    if not files_to_process:
        print("No new files to add. All specified files are already in the database.")
        return

    print(
        f"Will process {len(files_to_process)} new files: {', '.join(files_to_process)}"
    )

    # Prepare lists for new data
    new_chunks = []
    new_metadata = []

    # Process each new file
    print("\n1. PROCESSING NEW FILES\n")
    for i, file_path in enumerate(files_to_process):
        print(f"\nProcessing file {i+1}/{len(files_to_process)}: {file_path}")
        chunk_count = process_file_in_chunks(file_path, new_chunks, new_metadata)
        gc.collect()
        print(f"Memory cleanup completed after processing {file_path}")

    if not new_chunks:
        print("No new chunks were generated. Database not updated.")
        return

    print(f"\n2. GENERATING EMBEDDINGS FOR {len(new_chunks)} NEW CHUNKS\n")
    new_embeddings = get_embeddings(new_chunks)

    print("\n3. UPDATING FAISS INDEX\n")
    print(
        f"Adding {len(new_embeddings)} new vectors to existing index with {index.ntotal} vectors"
    )

    # Add new embeddings to the FAISS index
    index.add(new_embeddings)
    print(f"Index updated. New total: {index.ntotal} vectors")

    # Update metadata and chunks
    print("\n4. UPDATING METADATA\n")
    print(
        f"Adding {len(new_metadata)} new metadata entries to existing {len(metadata)} entries"
    )
    metadata.extend(new_metadata)
    chunks.extend(new_chunks)

    # Save the updated index and metadata
    print("\n5. SAVING UPDATED DATABASE\n")
    os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)
    print(f"Saving updated FAISS index to {VECTOR_DB_PATH}...")
    faiss.write_index(index, VECTOR_DB_PATH)

    print(f"Saving updated metadata to {VECTOR_DB_METADATA_PATH}...")
    with open(VECTOR_DB_METADATA_PATH, "wb") as f:
        pickle.dump((metadata, chunks), f)

    print("\n--- Vector Database Update Complete ---")
    print(f"Database now has {len(chunks)} chunks and {index.ntotal} vectors")


def build_vector_database() -> None:
    """Build a FAISS vector database from PDF files."""
    print("\n--- Starting Vector Database Build ---\n")

    if (
        os.path.exists(VECTOR_DB_PATH)
        and os.path.exists(VECTOR_DB_METADATA_PATH)
        and not args.force
    ):
        print(f"Vector database already exists at {VECTOR_DB_PATH}")
        overwrite = input("Do you want to rebuild it? (y/n): ").lower().strip()
        if overwrite != "y":
            print("Exiting without changes.")
            return
        print("Rebuilding vector database...")

    all_chunks = []
    all_metadata = []

    # Extract and chunk text from each file
    print("\n1. PROCESSING FILES\n")
    for i, file_path in enumerate(INPUT_FILES):
        print(f"\nProcessing file {i+1}/{len(INPUT_FILES)}: {file_path}")

        chunk_count = process_file_in_chunks(file_path, all_chunks, all_metadata)

        # Force garbage collection after each file
        gc.collect()
        print(f"Memory cleanup completed after processing {file_path}")
        print(f"Current total: {len(all_chunks)} chunks in database")

    # Get embeddings
    print("\n2. GENERATING EMBEDDINGS\n")
    if not all_chunks:
        print("No chunks were generated. Vector database not built.")
        return

    print(f"Starting embeddings generation for {len(all_chunks)} chunks...")
    embeddings = get_embeddings(all_chunks)

    # Build FAISS index
    print("\n3. BUILDING FAISS INDEX\n")
    dimension = embeddings.shape[1]
    print(f"Embedding dimension: {dimension}")

    index = faiss.IndexFlatL2(dimension)
    print(f"Created FAISS IndexFlatL2 with dimension {dimension}")

    print("Adding embeddings to index...")
    index.add(embeddings)
    print(f"Added {index.ntotal} vectors to index")

    # Save the index and metadata
    print("\n4. SAVING DATABASE\n")
    os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)
    print(f"Saving FAISS index to {VECTOR_DB_PATH}...")
    faiss.write_index(index, VECTOR_DB_PATH)

    print(f"Saving metadata to {VECTOR_DB_METADATA_PATH}...")
    with open(VECTOR_DB_METADATA_PATH, "wb") as f:
        pickle.dump((all_metadata, all_chunks), f)

    print("\n--- Vector Database Build Complete ---")
    print(f"Created database with {len(all_chunks)} chunks")
    print(f"Index saved to: {VECTOR_DB_PATH}")
    print(f"Metadata saved to: {VECTOR_DB_METADATA_PATH}")


if __name__ == "__main__":
    if args.mode == "build":
        build_vector_database()
    else:  # update mode
        update_vector_database(INPUT_FILES)
    print("\nDone!")
