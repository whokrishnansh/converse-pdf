"""
ConversePDF — Chat with your PDFs using Retrieval-Augmented Generation (RAG).

A single-file Streamlit application that lets users upload PDF documents,
index their content with FAISS vector search, and ask natural-language
questions answered by OpenAI's GPT model.

Author : Krishnansh Sharma
Stack  : Streamlit · LangChain · FAISS · PyPDF · OpenAI
"""

# ──────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────
import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


# ──────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="ConversePDF",
    page_icon="📄",
    layout="wide",
)


# ──────────────────────────────────────────────
# Custom CSS for a polished, portfolio-ready look
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---------- Header ---------- */
    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #888;
        margin-bottom: 2rem;
    }

    /* ---------- Answer card ---------- */
    .answer-card {
        background: linear-gradient(145deg, #1e1e2f, #2a2a40);
        border: 1px solid #3a3a5c;
        border-radius: 14px;
        padding: 1.6rem 2rem;
        margin-top: 1.2rem;
        color: #e0e0e0;
        line-height: 1.75;
        font-size: 1.02rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #c5c6f7;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #a0a0c0;
        font-size: 0.92rem;
    }

    /* ---------- Status badges ---------- */
    .status-ready {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        background: rgba(56, 219, 149, 0.15);
        color: #38db95;
        border: 1px solid rgba(56, 219, 149, 0.3);
    }
    .status-waiting {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        background: rgba(255, 193, 7, 0.12);
        color: #ffc107;
        border: 1px solid rgba(255, 193, 7, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Core helper functions
# ──────────────────────────────────────────────

def get_pdf_text(pdf_docs: list) -> str:
    """
    Extract raw text from a list of uploaded PDF files.

    Uses LangChain's PyPDFLoader to read each page of every uploaded PDF,
    then concatenates all page contents into a single string.

    Args:
        pdf_docs: List of Streamlit UploadedFile objects (PDF).

    Returns:
        A single string containing all text extracted from the PDFs.
    """
    text = ""
    for pdf in pdf_docs:
        # PyPDFLoader requires a file path, so we write the upload to a
        # temporary file and read from there.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        for page in pages:
            text += page.page_content

        # Clean up the temporary file after loading.
        os.unlink(tmp_path)

    return text


def get_text_chunks(text: str) -> list[str]:
    """
    Split a large body of text into overlapping chunks suitable for embedding.

    Uses LangChain's RecursiveCharacterTextSplitter with sensible defaults
    for RAG pipelines (10 000-character chunks with 1 000-character overlap).

    Args:
        text: The full document text.

    Returns:
        A list of text chunk strings.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10_000,
        chunk_overlap=1_000,
    )
    chunks = splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks: list[str], api_key: str) -> None:
    """
    Create a FAISS vector index from the provided text chunks and persist
    it to disk so it can be loaded later for similarity search.

    Args:
        text_chunks: List of text strings to embed.
        api_key: OpenAI API key for the embedding model.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def handle_user_question(user_question: str, api_key: str) -> str:
    """
    Given a user's natural-language question, perform similarity search on
    the persisted FAISS index and pass the top-matching chunks to the
    conversational chain to generate a grounded answer.

    Args:
        user_question: The question typed by the user.
        api_key: OpenAI API key.

    Returns:
        The model's answer string.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
    )

    # Load the previously saved FAISS index.
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # Retrieve the most relevant document chunks.
    docs = vector_store.similarity_search(user_question, k=5)

    # Combine retrieved chunks into a single context string.
    context = "\n\n".join(doc.page_content for doc in docs)

    # Build the prompt.
    prompt_template = PromptTemplate(
        template="""
    You are an expert document analyst. Answer the question as thoroughly
    and accurately as possible using **only** the provided context.

    If the answer is not contained in the context, respond with:
    "The answer is not available in the provided documents."

    Do **not** fabricate information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """,
        input_variables=["context", "question"],
    )

    # Build the model and invoke directly.
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        openai_api_key=api_key,
    )

    formatted_prompt = prompt_template.format(
        context=context,
        question=user_question,
    )

    response = model.invoke(formatted_prompt)
    return response.content


# ──────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────

def main():
    """Entry-point — renders the full Streamlit interface."""

    # ── Header ──────────────────────────────
    st.markdown('<p class="main-title">📄 ConversePDF</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">'
        "Upload your PDFs, build a knowledge base, and chat with your documents — "
        "powered by OpenAI &amp; FAISS vector search."
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Sidebar ─────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # API key input (masked).
        api_key = st.text_input(
            "🔑 OpenAI API Key",
            type="password",
            help="Get yours at https://platform.openai.com/api-keys",
        )

        st.markdown("---")
        st.markdown("## 📂 Upload Documents")

        pdf_docs = st.file_uploader(
            "Choose one or more PDF files",
            type="pdf",
            accept_multiple_files=True,
        )

        process_btn = st.button("🚀 Submit & Process", use_container_width=True)

        # ── Processing logic ────────────────
        if process_btn:
            if not api_key:
                st.error("Please enter your OpenAI API Key above.")
            elif not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("📖 Extracting text from PDFs…"):
                    raw_text = get_pdf_text(pdf_docs)

                if not raw_text.strip():
                    st.error(
                        "No readable text found in the uploaded PDFs. "
                        "They may be scanned images — try OCR-processed files."
                    )
                else:
                    with st.spinner("✂️ Splitting text into chunks…"):
                        text_chunks = get_text_chunks(raw_text)

                    with st.spinner("🧠 Creating vector embeddings…"):
                        get_vector_store(text_chunks, api_key)

                    st.success(
                        f"✅ Done! Indexed **{len(text_chunks)}** chunks from "
                        f"**{len(pdf_docs)}** PDF(s). You can now ask questions."
                    )
                    # Persist a flag so the main area knows the index is ready.
                    st.session_state["index_ready"] = True

        # ── Sidebar status indicator ────────
        st.markdown("---")
        if st.session_state.get("index_ready"):
            st.markdown('<span class="status-ready">● Index Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-waiting">● Awaiting PDFs</span>', unsafe_allow_html=True)

        # ── How-to section ──────────────────
        st.markdown("---")
        st.markdown("## 💡 How to use")
        st.markdown(
            """
            1. Paste your **OpenAI API Key**.
            2. Upload one or more **PDF** files.
            3. Click **Submit & Process**.
            4. Type a question in the main area.
            """
        )

    # ── Main content area ───────────────────
    st.markdown("### 💬 Ask a Question")

    user_question = st.text_input(
        "Type your question about the uploaded documents:",
        placeholder="e.g. What are the key findings in Chapter 3?",
        label_visibility="collapsed",
    )

    if user_question:
        if not api_key:
            st.warning("⚠️ Please enter your OpenAI API Key in the sidebar.")
        elif not st.session_state.get("index_ready"):
            st.warning("⚠️ Please upload and process your PDFs first.")
        else:
            with st.spinner("🔍 Searching documents & generating answer…"):
                try:
                    answer = handle_user_question(user_question, api_key)
                    st.markdown(
                        f'<div class="answer-card">{answer}</div>',
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Something went wrong: {e}")


# ──────────────────────────────────────────────
# Run the app
# ──────────────────────────────────────────────
if __name__ == "__main__":
    main()
