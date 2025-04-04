# NLP Track  Emotions - Streamlit Web App

This repository contains the source code for the Streamlit web application, which explores whether natural language processing (NLP) can track negative emotions in adolescents' daily lives.

## Project Structure

**pages/** - Contains the code for all the web application pages.

**data/** - Contains the CSV files used for generating graphs.

**docs/** - Stores document-based data that serves as the knowledge base for the chatbot.

**hadar_faiss_index/** - Contains the embeddings for the chatbot using free models (not in use currently as we have switched to LightRAG with OpenAI LLM).

**new_lightrag_working_dir/** - The active directory containing embeddings and knowledge for the LightRAG chatbot, currently not in use.

**FREE_MODELS_app_with_chatbot.py** - Code for the chatbot using free models (via OpenRouter.ai), but not in use currently.

**new_app_chatbot.py** - Code for the chatbot, which is currently being used. it uses file_search tool with the pdf files from the docs folder

**threed.html** - Contains the code for the bird animation effect displayed on the main page of the app.

**website_lightrag.py** - Coontains the lightrag cli code, use this file with the code in line 186 to add more docs and create new lightrag db

**new_app_chatbot_bak.py** - Code for the LightRAG chatbot, which is not currently being used.

## Getting Started

To run the web app locally, follow these steps:

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

**Dependencies**

Ensure you have Python >=3.10 installed and the necessary packages listed in requirements.txt.
