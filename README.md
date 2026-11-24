# AI Document Assistant

An AI-powered PDF question-answering application built using Retrieval-Augmented Generation (RAG). Upload a PDF, ask questions about its content, and receive AI-generated answers with source-page references.

## Features

- PDF text extraction using PyMuPDF
- Sentence-aware text chunking
- Text embeddings using Sentence Transformers
- Semantic search using FAISS
- Context-aware answers using Google Gemini API
- Document summarisation
- Source-page references
- Interactive Streamlit interface
- Response-time tracking and API error handling

## Technology Stack

- Python
- Streamlit
- Google Gemini API
- FAISS
- Sentence Transformers
- PyMuPDF
- NumPy

## Installation

1. Clone the repository:

   git clone YOUR_GITHUB_REPOSITORY_URL

2. Navigate into the project folder:

   cd AI-Document-Assistant

3. Create a virtual environment:

   python -m venv .venv

4. Activate it on Windows:

   .venv\Scripts\activate

5. Install dependencies:

   pip install -r requirements.txt

6. Create a `.env` file in the project root and add:

   GEMINI_API_KEY=your_gemini_api_key

7. Run the application:

   streamlit run app.py

## Security

Never upload your `.env` file or expose your Gemini API key.

## Project Status

Developed as a hands-on Generative AI and RAG project.

