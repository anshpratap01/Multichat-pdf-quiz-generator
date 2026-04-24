# Multi PDF Chat and Quiz Generator

A Streamlit application that lets you:
- Upload one or more PDF files
- Ask questions over the uploaded content (RAG style)
- Generate multiple-choice quizzes from selected PDF content

The app uses:
- Groq API for chat and quiz generation
- FAISS for vector similarity search
- sentence-transformers embeddings (with fallback hashing embedder)
- PyMuPDF for PDF text extraction

## Project Structure

app.py
requirements.txt
utils/
  groq_client.py
  pdf_processor.py
  vector_store.py

## Features

- Multi-PDF upload and indexing
- Context-based chat with source citations
- Quiz generation from one PDF or all PDFs
- Score calculation and answer explanations
- Fallback model handling for Groq model changes
- Fallback embedding strategy when sentence-transformers stack fails

## Prerequisites

- Python 3.10+ (tested with Python 3.12)
- Git (optional, for version control)
- Groq API key

## Installation

1. Clone the repository:

   git clone https://github.com/anshpratap01/Multichat-pdf-quiz-generator.git
   cd Multichat-pdf-quiz-generator

2. Create and activate virtual environment (Windows PowerShell):

   python -m venv .venv
   .venv\Scripts\Activate.ps1

3. Install dependencies:

   pip install -r requirements.txt

## Environment Setup

Create a file named .env in the project root with:

GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant

Notes:
- Keep .env private and never commit it.
- Use .env.example as a safe template for sharing.

## Run the App

Start Streamlit:

streamlit run app.py

Then open the local URL shown in terminal (usually http://localhost:8501).

## How to Use

1. Upload one or more PDF files from the sidebar.
2. Click Process PDFs.
3. Use Chat with PDFs tab to ask questions.
4. Use Generate Quiz tab to create and submit an MCQ quiz.

## Troubleshooting

### ModuleNotFoundError for torchvision

Install it in your active virtual environment:

pip install torchvision

If needed, reinstall torch and torchvision together:

pip install --upgrade torch torchvision

### Git push blocked due to secret scanning

If push is blocked because .env was committed:

1. Remove secrets from tracked files and history.
2. Ensure .env is listed in .gitignore.
3. Commit clean files and push again.
4. Rotate the leaked API key immediately.

### Slow first run

The first embedding model load can take extra time while dependencies initialize.

## Security Notes

- Do not commit .env or API keys.
- Rotate keys immediately if they were ever pushed to a remote repository.
- Use .env.example for shared configuration templates.

## Suggested Next Improvements

- Add persistent FAISS index storage
- Add PDF page-level citations
- Add export for quiz results
- Add Docker support for easier deployment

## License

Add your preferred license information here.
