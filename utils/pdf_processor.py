"""
pdf_processor.py
----------------
Handles PDF text extraction using PyMuPDF (fitz) and text chunking
using a lightweight internal splitter.

Each chunk is returned as a dict with:
  - 'text': the chunk content
  - 'source': the original PDF filename
"""

import fitz  # PyMuPDF


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract all text from an uploaded Streamlit file object using PyMuPDF.
    Reads the file bytes directly — no temp file needed.
    Returns the full concatenated text of all pages.
    """
    # Read raw bytes from the Streamlit UploadedFile
    pdf_bytes = uploaded_file.read()

    # Open in-memory — fitz accepts a bytes stream
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    full_text = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text("text")  # plain text extraction
        if page_text.strip():
            full_text.append(page_text)

    doc.close()
    return "\n".join(full_text)


def chunk_text(text: str, source_name: str) -> list[dict]:
    """
    Split a long text string into overlapping chunks using LangChain splitter.

    Args:
        text: Full document text.
        source_name: The PDF filename, stored as metadata on each chunk.

    Returns:
        List of dicts: [{'text': '...', 'source': 'file.pdf'}, ...]
    """
    chunk_size = 500
    chunk_overlap = 50
    separators = ["\n\n", "\n", ". ", " "]

    raw_chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            # Prefer a natural break close to the chunk boundary.
            window = text[start:end]
            best_break = -1
            for sep in separators:
                idx = window.rfind(sep)
                if idx > best_break:
                    best_break = idx

            if best_break > int(chunk_size * 0.5):
                end = start + best_break + 1

        chunk = text[start:end].strip()
        if chunk:
            raw_chunks.append(chunk)

        if end >= text_len:
            break

        next_start = end - chunk_overlap
        start = next_start if next_start > start else end

    # Tag every chunk with its source filename for citation in responses
    return [{"text": chunk, "source": source_name} for chunk in raw_chunks]


def process_uploaded_pdfs(uploaded_files) -> list[dict]:
    """
    Full pipeline: take a list of Streamlit UploadedFile objects,
    extract text from each, chunk them, and return all chunks combined.

    Args:
        uploaded_files: list of st.UploadedFile objects

    Returns:
        Flat list of chunk dicts from all PDFs combined.
    """
    all_chunks = []

    for uploaded_file in uploaded_files:
        # Reset file pointer in case it was read before
        uploaded_file.seek(0)

        source_name = uploaded_file.name

        # Extract raw text from this PDF
        text = extract_text_from_pdf(uploaded_file)

        if not text.strip():
            # Empty PDF — skip gracefully
            continue

        # Chunk and tag with source filename
        chunks = chunk_text(text, source_name)
        all_chunks.extend(chunks)

    return all_chunks
