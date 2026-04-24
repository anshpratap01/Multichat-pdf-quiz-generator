"""
app.py
------
Multi-PDF Chat + Quiz Generator
Streamlit app powered by Groq (llama3-8b-8192), FAISS, and sentence-transformers.

Run with:
    streamlit run app.py
"""

import streamlit as st
from utils.pdf_processor import process_uploaded_pdfs
from utils.vector_store import build_faiss_index
from utils.groq_client import chat_with_docs, generate_quiz

# ─────────────────────────────────────────────
# Page config — must be the FIRST Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-PDF Chat + Quiz Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — clean, readable dark-accent theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d27 0%, #151820 100%);
        border-right: 1px solid #2d3148;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1d27;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8892b0;
        font-weight: 600;
        font-size: 15px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2d3148 !important;
        color: #64ffda !important;
        border-radius: 6px;
    }

    /* Source pill badge */
    .source-badge {
        display: inline-block;
        background: #1e2a3a;
        border: 1px solid #2d4a6a;
        color: #5fa8d3;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 12px;
        margin: 3px 3px 0 0;
    }

    /* Quiz question card */
    .quiz-card {
        background: #1a1d27;
        border: 1px solid #2d3148;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .quiz-question {
        font-weight: 700;
        font-size: 15px;
        color: #ccd6f6;
        margin-bottom: 10px;
    }

    /* Score display */
    .score-box {
        background: linear-gradient(135deg, #0d2137, #0d3726);
        border: 1px solid #64ffda44;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
    }
    .score-number { font-size: 48px; font-weight: 900; color: #64ffda; }
    .score-label  { font-size: 14px; color: #8892b0; margin-top: 4px; }

    /* Result indicators */
    .correct-answer   { color: #64ffda; font-weight: 700; }
    .incorrect-answer { color: #ff6b6b; font-weight: 700; }
    .explanation-text { color: #8892b0; font-size: 13px; margin-top: 6px; font-style: italic; }

    /* Chat tweaks */
    [data-testid="stChatMessage"] {
        background: #1a1d27;
        border: 1px solid #2d3148;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session State Initialisation
# ─────────────────────────────────────────────
def init_session_state():
    """Set default values for all session state keys on first run."""
    defaults = {
        "chat_history": [],           # list of {role, content, sources}
        "faiss_store": None,          # VectorStore instance
        "all_chunks": [],             # flat list of all chunk dicts
        "uploaded_file_names": [],    # list of PDF filenames processed
        "quiz_data": [],              # list of question dicts
        "quiz_submitted": False,      # whether the user has submitted answers
        "user_answers": {},           # {question_index: selected_option}
        "pdf_text_map": {},           # {filename: full_text} for quiz selection
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# ─────────────────────────────────────────────
# SIDEBAR — PDF upload & processing
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 PDF Chat + Quiz")
    st.markdown("---")

    # ── File uploader ──
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDFs to chat with or quiz on.",
    )

    # ── Process button ──
    if st.button("⚡ Process PDFs", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF first.")
        else:
            with st.spinner("Extracting text and building index…"):
                try:
                    # Extract + chunk all PDFs
                    all_chunks = process_uploaded_pdfs(uploaded_files)

                    if not all_chunks:
                        st.error("No text could be extracted from the uploaded PDFs.")
                    else:
                        # Build FAISS vector index
                        faiss_store = build_faiss_index(all_chunks)

                        # Build a {filename: full_text} map for quiz generation
                        pdf_text_map = {}
                        for chunk in all_chunks:
                            src = chunk["source"]
                            pdf_text_map[src] = pdf_text_map.get(src, "") + " " + chunk["text"]

                        # Persist to session state
                        st.session_state["faiss_store"] = faiss_store
                        st.session_state["all_chunks"] = all_chunks
                        st.session_state["uploaded_file_names"] = [f.name for f in uploaded_files]
                        st.session_state["pdf_text_map"] = pdf_text_map

                        # Reset quiz & chat when new PDFs are loaded
                        st.session_state["quiz_data"] = []
                        st.session_state["quiz_submitted"] = False
                        st.session_state["user_answers"] = {}
                        st.session_state["chat_history"] = []

                        st.success(
                            f"✅ Indexed **{len(all_chunks)} chunks** "
                            f"from **{len(uploaded_files)} PDF(s)**"
                        )

                except Exception as e:
                    st.error(f"Error processing PDFs: {e}")

    # ── Status indicator ──
    if st.session_state["uploaded_file_names"]:
        st.markdown("**Loaded PDFs:**")
        for name in st.session_state["uploaded_file_names"]:
            st.markdown(f"• `{name}`")

    st.markdown("---")

    # ── Quiz question count control ──
    num_quiz_questions = st.number_input(
        "Quiz questions",
        min_value=5,
        max_value=15,
        value=10,
        step=1,
        help="Number of MCQ questions to generate in the Quiz tab.",
    )

    st.markdown("---")
    st.caption("Powered by Groq · FAISS · sentence-transformers")


# ─────────────────────────────────────────────
# MAIN AREA — Two tabs
# ─────────────────────────────────────────────
tab_chat, tab_quiz = st.tabs(["💬  Chat with PDFs", "🧠  Generate Quiz"])


# ══════════════════════════════════════════════
# TAB 1 — CHAT WITH PDFs
# ══════════════════════════════════════════════
with tab_chat:
    st.markdown("### 💬 Chat with your PDFs")

    if not st.session_state["faiss_store"]:
        st.info("👈 Upload and process PDFs using the sidebar to start chatting.")
    else:
        # ── Render existing chat history ──
        for message in st.session_state["chat_history"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Show source badges below assistant messages
                if message["role"] == "assistant" and message.get("sources"):
                    unique_sources = list(dict.fromkeys(message["sources"]))
                    badges = "".join(
                        f'<span class="source-badge">📄 {s}</span>'
                        for s in unique_sources
                    )
                    st.markdown(
                        f'<div style="margin-top:8px">{badges}</div>',
                        unsafe_allow_html=True,
                    )

        # ── Chat input ──
        user_query = st.chat_input("Ask a question about your documents…")

        if user_query:
            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(user_query)

            # Save user message to history
            st.session_state["chat_history"].append(
                {"role": "user", "content": user_query, "sources": []}
            )

            # Retrieve top-4 relevant chunks from FAISS
            with st.spinner("Searching documents…"):
                try:
                    relevant_chunks = st.session_state["faiss_store"].search(
                        user_query, k=4
                    )

                    # Call Groq with context
                    answer = chat_with_docs(user_query, relevant_chunks)
                    source_names = [c["source"] for c in relevant_chunks]

                    # Display assistant response
                    with st.chat_message("assistant"):
                        st.markdown(answer)

                        unique_sources = list(dict.fromkeys(source_names))
                        badges = "".join(
                            f'<span class="source-badge">📄 {s}</span>'
                            for s in unique_sources
                        )
                        st.markdown(
                            f'<div style="margin-top:8px">{badges}</div>',
                            unsafe_allow_html=True,
                        )

                    # Save assistant message to history
                    st.session_state["chat_history"].append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": source_names,
                        }
                    )

                except Exception as e:
                    st.error(f"Error getting response: {e}")


# ══════════════════════════════════════════════
# TAB 2 — GENERATE QUIZ
# ══════════════════════════════════════════════
with tab_quiz:
    st.markdown("### 🧠 Generate a Quiz")

    if not st.session_state["faiss_store"]:
        st.info("👈 Upload and process PDFs using the sidebar to generate a quiz.")
    else:
        file_names = st.session_state["uploaded_file_names"]
        pdf_text_map = st.session_state["pdf_text_map"]

        # ── PDF selection dropdown ──
        quiz_source_options = ["All PDFs"] + file_names
        selected_source = st.selectbox(
            "Quiz source",
            options=quiz_source_options,
            help="Choose which PDF to quiz on, or use all combined.",
        )

        # ── Generate quiz button ──
        if st.button("🎯 Generate Quiz", type="primary"):
            # Build the topic text for the selected source
            if selected_source == "All PDFs":
                topic_text = " ".join(pdf_text_map.values())
            else:
                topic_text = pdf_text_map.get(selected_source, "")

            if not topic_text.strip():
                st.error("No text found for the selected PDF(s).")
            else:
                with st.spinner("Generating quiz questions… this may take 15–30 seconds."):
                    try:
                        quiz = generate_quiz(topic_text, num_questions=num_quiz_questions)

                        # Reset quiz state for new quiz
                        st.session_state["quiz_data"] = quiz
                        st.session_state["quiz_submitted"] = False
                        st.session_state["user_answers"] = {}

                        st.success(f"✅ Generated {len(quiz)} questions from **{selected_source}**")

                    except ValueError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Quiz generation failed: {e}")

        # ── Render quiz if available ──
        if st.session_state["quiz_data"]:
            quiz = st.session_state["quiz_data"]
            submitted = st.session_state["quiz_submitted"]

            st.markdown("---")

            # ── Score summary (shown after submission) ──
            if submitted:
                correct_count = 0
                for i, q in enumerate(quiz):
                    if st.session_state["user_answers"].get(i) == q["answer"]:
                        correct_count += 1

                pct = int((correct_count / len(quiz)) * 100)
                emoji = "🎉" if pct >= 80 else "📖" if pct >= 50 else "💪"

                st.markdown(
                    f"""
                    <div class="score-box">
                        <div class="score-number">{emoji} {correct_count}/{len(quiz)}</div>
                        <div class="score-label">Score: {pct}% — 
                        {"Excellent!" if pct >= 80 else "Good effort!" if pct >= 50 else "Keep studying!"}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ── Render each question ──
            with st.form(key="quiz_form"):
                for i, q in enumerate(quiz):
                    st.markdown(
                        f'<div class="quiz-card">'
                        f'<div class="quiz-question">Q{i+1}. {q["question"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    options = q["options"]
                    option_labels = [f"{k}: {v}" for k, v in options.items()]

                    # Determine default index (restore previous answer or None)
                    prev_answer = st.session_state["user_answers"].get(i)
                    option_keys = list(options.keys())
                    default_idx = option_keys.index(prev_answer) if prev_answer in option_keys else 0

                    selected = st.radio(
                        label=f"q_{i}",           # hidden label
                        options=option_labels,
                        index=default_idx,
                        key=f"quiz_q_{i}",
                        label_visibility="collapsed",
                        disabled=submitted,       # lock after submission
                    )

                    # Show result feedback after submission
                    if submitted:
                        user_key = st.session_state["user_answers"].get(i, "")
                        correct_key = q["answer"]
                        correct_label = f"{correct_key}: {options[correct_key]}"

                        if user_key == correct_key:
                            st.markdown(
                                f'<div class="correct-answer">✅ Correct!</div>'
                                f'<div class="explanation-text">💡 {q.get("explanation", "")}</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f'<div class="incorrect-answer">❌ Incorrect. '
                                f'Correct answer: {correct_label}</div>'
                                f'<div class="explanation-text">💡 {q.get("explanation", "")}</div>',
                                unsafe_allow_html=True,
                            )

                    st.markdown("")  # visual spacing between questions

                # ── Submit button ──
                if not submitted:
                    submit_clicked = st.form_submit_button(
                        "📝 Submit Answers",
                        type="primary",
                        use_container_width=True,
                    )

                    if submit_clicked:
                        # Collect all answers from form widgets
                        answers = {}
                        for i, q in enumerate(quiz):
                            widget_val = st.session_state.get(f"quiz_q_{i}", "")
                            # Extract just the letter key (e.g. "A" from "A: Some text")
                            selected_key = widget_val.split(":")[0].strip() if widget_val else ""
                            answers[i] = selected_key

                        st.session_state["user_answers"] = answers
                        st.session_state["quiz_submitted"] = True
                        st.rerun()
                else:
                    # "Try Again" button resets quiz state
                    retry_clicked = st.form_submit_button(
                        "🔄 Try Again",
                        use_container_width=True,
                    )
                    if retry_clicked:
                        st.session_state["quiz_submitted"] = False
                        st.session_state["user_answers"] = {}
                        st.rerun()
