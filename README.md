# NLP Track Emotions - Streamlit Web App

This repository contains the source code for the Streamlit web application, which explores whether natural language processing (NLP) can track negative emotions in adolescents daily lives.

## Project Structure

**pages/** - Contains the code for all the web application pages.

**data/** - Contains the CSV files used for generating graphs.

**docs/** - Stores document-based data that serves as the knowledge base for the chatbot.

**hadar_faiss_index/** - Contains the embeddings for the chatbot using free models (not in use currently as we have switched to gpt-4.1-mini with  OpenAI embeddings).

**new_lightrag_working_dir/** - The active directory containing embeddings and knowledge for the LightRAG chatbot, currently not in use.

**FREE_MODELS_app_with_chatbot.py** - Code for the chatbot using free models (via OpenRouter.ai), but not in use currently.

**new_app_chatbot.py** - Code for the chatbot, which is currently being used. It uses the `file_search` tool with the PDF files from the `docs/` folder.

**build_vector_db.py** is a command-line script that you must run locally before using the chatbot with your vector store files. Please refer to the notes at the bottom of the page.

**threed.html** - Contains the code for the bird animation effect displayed on the main page of the app.

**website_lightrag.py** - Contains the LightRAG CLI code; use this file with the code near line 186 to add more documents and create a new LightRAG database.

**new_app_chatbot_bak.py** - Backup code for the LightRAG chatbot, not currently in use.

## Getting Started

1. Clone the repository:
   ```sh
   git clone https://github.com/yweiss0/Hadar_Fisher_webapp.git
   cd Hadar_Fisher_webapp
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```

## Dependencies

Ensure you have Python >=3.10 installed and the necessary packages listed in `requirements.txt`.

## build_vector_db.py

`build_vector_db.py` is a command-line script that you must run locally before using the chatbot with your vector store files. It performs the following steps:

- Extracts text from PDF and TXT files in the `docs/` directory.
- Sanitizes the extracted text to handle encoding issues and remove invalid characters.
- Splits documents into overlapping text chunks for more effective retrieval.
- Generates embeddings for each chunk using OpenAI's embeddings API.
- Builds a FAISS index and saves both the index (`docs/vector_store.faiss`) and metadata (`docs/vector_store_metadata.pkl`).

To build the initial vector database, run:
```sh
python build_vector_db.py --mode build
```

For detailed usage, configuration options, and advanced modes (e.g., `update`), please see the separate README at `build_vector_db_README.md`. 
