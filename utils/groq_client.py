"""
groq_client.py
--------------
Wraps the Groq Python SDK for two purposes:
  1. chat_with_docs — RAG-style Q&A using retrieved context chunks
  2. generate_quiz — MCQ quiz generation from topic text

Model selection is configurable via GROQ_MODEL with safe fallbacks.
"""

import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

# Load GROQ_API_KEY from .env file
load_dotenv()

# Default model can be overridden from .env using GROQ_MODEL
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
FALLBACK_GROQ_MODELS = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]


def get_groq_client() -> Groq:
    """
    Initialize and return a Groq client.
    Raises a clear error if the API key is missing.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Please add it to your .env file: GROQ_API_KEY=your_key_here"
        )
    return Groq(api_key=api_key)


def _get_candidate_models() -> list[str]:
    """Build an ordered, de-duplicated model list for retry/fallback."""
    seen = set()
    candidates = [DEFAULT_GROQ_MODEL, *FALLBACK_GROQ_MODELS]
    ordered = []
    for model in candidates:
        if model and model not in seen:
            seen.add(model)
            ordered.append(model)
    return ordered


def _create_chat_completion(client: Groq, messages: list[dict], temperature: float, max_tokens: int):
    """Call Groq with fallback models when a model was retired/decommissioned."""
    last_error = None

    for model_name in _get_candidate_models():
        try:
            return client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            last_error = e
            error_text = str(e).lower()
            # Try the next model only for known model selection failures.
            if (
                "model_decommissioned" in error_text
                or "decommissioned" in error_text
                or "not found" in error_text
                or "invalid model" in error_text
            ):
                continue
            raise

    raise ValueError(
        "All configured Groq models failed. "
        "Set GROQ_MODEL in your .env to a currently supported model. "
        f"Last error: {last_error}"
    )


def chat_with_docs(query: str, context_chunks: list[dict]) -> str:
    """
    Answer a user question using retrieved document chunks as context.

    Args:
        query: The user's question.
        context_chunks: List of {'text': ..., 'source': ...} dicts
                        from VectorStore.search().

    Returns:
        The assistant's answer as a plain string.
    """
    client = get_groq_client()

    # Build the context block from retrieved chunks
    context_text = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )

    system_prompt = (
        "You are a helpful assistant that answers questions strictly based on "
        "the provided document context. If the answer is not in the context, "
        "say so clearly. Be concise and accurate. Always cite which document "
        "your answer comes from when relevant."
    )

    user_message = (
        f"Context from uploaded documents:\n\n{context_text}\n\n"
        f"Question: {query}"
    )

    response = _create_chat_completion(
        client=client,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,  # lower temp = more factual, less creative
        max_tokens=1024,
    )

    return response.choices[0].message.content


def generate_quiz(topic_text: str, num_questions: int = 10) -> list[dict]:
    """
    Generate a multiple-choice quiz from the given topic text.

    Args:
        topic_text: Combined text from the selected PDF(s).
        num_questions: How many MCQ questions to generate (5–15).

    Returns:
        Parsed list of question dicts:
        [
          {
            "question": "...",
            "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
            "answer": "A",
            "explanation": "..."
          },
          ...
        ]

    Raises:
        ValueError: If Groq returns malformed JSON.
    """
    client = get_groq_client()

    # Truncate topic text to avoid hitting context limits
    # ~12 000 chars ≈ 3 000 tokens — leaves room for the response
    truncated_text = topic_text[:12000]

    system_prompt = (
        "You are an expert quiz generator. "
        "You MUST respond with ONLY a valid JSON array — no markdown, "
        "no explanations, no code fences. "
        "The JSON must be parseable by Python's json.loads()."
    )

    user_message = f"""Generate exactly {num_questions} multiple-choice questions based on the text below.

Rules:
- Each question must have exactly 4 options: A, B, C, D
- Exactly one option is correct
- Questions should test comprehension, not trivia
- The "answer" field must be one of: "A", "B", "C", or "D"
- Include a brief explanation for the correct answer

Return ONLY this JSON structure (no extra text):
[
  {{
    "question": "Question text here?",
    "options": {{"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}},
    "answer": "A",
    "explanation": "Brief explanation of why A is correct."
  }}
]

Text to base questions on:
{truncated_text}"""

    response = _create_chat_completion(
        client=client,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.5,   # moderate creativity for varied questions
        max_tokens=4096,   # quiz responses can be long
    )

    raw_text = response.choices[0].message.content.strip()

    # Attempt to extract JSON even if the model added surrounding text
    json_match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if json_match:
        raw_text = json_match.group(0)

    try:
        quiz_data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Groq returned invalid JSON for quiz generation.\n"
            f"Raw response (first 500 chars):\n{raw_text[:500]}\n"
            f"JSON error: {e}"
        )

    # Validate basic structure
    validated = []
    for item in quiz_data:
        if all(k in item for k in ("question", "options", "answer", "explanation")):
            validated.append(item)

    if not validated:
        raise ValueError("Quiz generation returned no valid questions.")

    return validated
