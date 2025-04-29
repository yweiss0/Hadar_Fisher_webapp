# Vector Database for Chatbot RAG System

This repository contains a chatbot system that uses Retrieval-Augmented Generation (RAG) with a local vector database to answer questions based on documents.

## Key Files

- `build_vector_db.py`: Tool for building and updating the vector database
- `docs/`: Folder containing files and the vector database

## Supported File Types

- PDF files (`.pdf`)
- Text files (`.txt`)

## Setting Up

1. Install required dependencies:
```bash
pip install streamlit openai faiss-cpu PyPDF2 python-dotenv tqdm
```

2. Create a `.env` file in the project root with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

3. Build the initial vector database:
```bash
python build_vector_db.py --mode build
```

## How the Vector Database Works

The `build_vector_db.py` script creates a searchable vector database from your documents using the following process:

### 1. Text Extraction
- **PDF files**: Extracts text from each page using PyPDF2
- **Text files**: Reads content with appropriate encoding (UTF-8 or Latin-1)

### 2. Text Sanitization
- Handles problematic characters that could cause UTF-8 encoding issues
- Replaces surrogate characters and other invalid Unicode
- Preserves meaningful content while ensuring compatibility

### 3. Chunking
- Divides documents into smaller overlapping chunks (default: 800 characters with 150 overlap)
- Attempts to break chunks at natural points like sentence endings
- Each chunk is sanitized to prevent encoding issues

### 4. Embedding Generation
- Generates embeddings for each chunk using OpenAI's embeddings API
- Includes fallback sanitization if initial embedding fails
- Uses "text-embedding-3-small" model by default

### 5. Vector Database Creation
- Creates a FAISS index (Facebook AI Similarity Search)
- Stores both the vector index and metadata (source documents, chunk positions)
- Saves database to `docs/vector_store.faiss` and `docs/vector_store_metadata.pkl`

The script includes memory optimization to handle large documents and provides detailed progress updates during processing.

## Managing the Vector Database

### Building the Database

To build or rebuild the entire database:

```bash
python build_vector_db.py --mode build
```

Use the `--force` flag to rebuild without confirmation:

```bash
python build_vector_db.py --mode build --force
```

### Adding New Files

To add new files to the existing database without re-processing already embedded files:

```bash
python build_vector_db.py --mode update --files "path/to/new_file1.pdf" "path/to/new_file2.txt"
```

This will:
1. Load the existing vector database
2. Process only the new files
3. Add the new embeddings to the existing database
4. Save the updated database

### Error Handling

The script includes multiple safeguards:
- Sanitization of text to handle Unicode and encoding issues
- Fallback to ASCII-only content when necessary
- Memory management for large documents
- Skip processing for empty or corrupt files

### How it works inside the chatbot

The chatbot will automatically use the vector database for answering questions by:
1. Converting user queries into embeddings
2. Finding similar chunks in the vector database
3. Using the retrieved chunks as context for generating responses

## Command-Line Options

### build_vector_db.py

- `--mode [build|update]`: Choose between full database rebuild or adding new files
  - `build`: Process all files in the default list (or override with `--files`)
  - `update`: Add only new files to existing database
- `--files`: Specify files to process (space-separated list)
  - In `build` mode: overrides default file list
  - In `update` mode: specifies which new files to add
- `--force`: Skip confirmation when rebuilding an existing database 