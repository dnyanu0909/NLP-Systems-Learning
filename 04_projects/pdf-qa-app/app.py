import streamlit as st

from utils import (
    extract_text_from_pdf,
    smart_chunk,
    create_faiss_index,
    retrieve_answer,
    detect_query_type,
    summarize_document,
    generate_answer
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI PDF Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

# ---------------------------
# TITLE
# ---------------------------

st.title("📚 AI PDF Knowledge Assistant")

st.markdown("""
Ask intelligent questions from your PDF documents using:

- 🔍 Semantic Search
- 🧠 Transformers
- ⚡ FAISS Vector Database
- 🤖 Local LLMs (TinyLlama/Ollama)
""")

# ---------------------------
# SIDEBAR
# ---------------------------

with st.sidebar:

    st.header("🧠 About Project")

    st.write("""
This project is a Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents and ask intelligent questions about them.

The system combines:
- Semantic Retrieval
- Vector Search (FAISS)
- Transformer Embeddings
- Local LLM Generation
""")

    st.header("⚙️ Tech Stack")

    st.markdown("""
- Python
- Streamlit
- Sentence Transformers
- FAISS
- Ollama
- TinyLlama
""")

    st.header("🚀 Features")

    st.markdown("""
- PDF Upload
- Semantic Search
- AI Answer Generation
- Context Retrieval
- Local LLM Support
""")

# =========================================================
# PDF UPLOAD
# =========================================================
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

# =========================================================
# PROCESS PDF
# =========================================================
if uploaded_file is not None:

    # Spinner for better UX
    with st.spinner("Processing PDF..."):

        # Extract text
        text = extract_text_from_pdf(uploaded_file)

        # Smart chunking
        chunks = smart_chunk(text)

        # Create FAISS index
        index = create_faiss_index(chunks)

    st.success("✅ PDF processed successfully!")

    # =====================================================
    # USER QUERY
    # =====================================================
    query = st.text_input(
        "Ask a question about the PDF:"
    )

    # =====================================================
    # HANDLE QUERY
    # =====================================================
    if query:

        # Detect query type
        query_type = detect_query_type(query)

        # =================================================
        # SUMMARY MODE
        # =================================================
        if query_type == "summary":

            summary = summarize_document(text)

            st.subheader("📘 Document Summary")

            st.write(summary)

        # =================================================
        # RETRIEVAL QA MODE
        # =================================================
        else:

            with st.spinner("Searching document..."):

                retrieval_result = retrieve_answer(
                    query,
                    chunks,
                    index
                )

                if isinstance(retrieval_result, tuple) and len(retrieval_result) == 2:
                    retrieved_chunks, top_score = retrieval_result
                else:
                    retrieved_chunks = retrieval_result
                    top_score = None

                if top_score is not None:
                    st.metric("📊 Confidence Score", f"{top_score:.2f}")

                # Generate final answer
                final_answer = generate_answer(
                    query,
                    retrieved_chunks
                )

            # =============================================
            # SHOW RETRIEVED CONTEXT
            # =============================================
            st.subheader("📌 Retrieved Context")

            for i, chunk in enumerate(retrieved_chunks, start=1):

                with st.expander(f"Context Chunk {i}"):

                    st.write(chunk)

            # =============================================
            # FINAL AI ANSWER
            # =============================================
            st.markdown("---")
            
            st.subheader("🤖 AI Generated Answer")

            st.write(final_answer)

st.markdown("---")

st.caption("Built with ❤️ using NLP, RAG, Transformers, and Local LLMs")
# VERY Important Future Improvement

# Right now every query rebuilds nothing, which is good.

# BUT:

# Every page refresh reprocesses the whole PDF again.

# Bad for performance.

# You should cache:

# @st.cache_resource

# Example:

# @st.cache_resource
# def process_pdf(uploaded_file):

#     text = extract_text_from_pdf(uploaded_file)

#     chunks = smart_chunk(text)

#     index = create_faiss_index(chunks)

#     return text, chunks, index

# Then:

# text, chunks, index = process_pdf(uploaded_file)

# This is a BIG optimization.

# Another Huge Upgrade Later

# Currently your app:

# PDF → chunks → retrieve → answer

# Good beginner RAG.

# But advanced RAG adds:

# conversation memory
# hybrid search
# reranking models
# metadata filtering
# citations
# page number tracking
# chunk source highlighting

# You are already close to real AI engineering territory now.