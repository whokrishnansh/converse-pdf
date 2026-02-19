# 📄 ConversePDF

**Chat with your PDFs using Retrieval-Augmented Generation (RAG).**

ConversePDF is a Streamlit-powered app that lets you upload PDF documents, index their content with FAISS vector search, and ask natural-language questions — answered accurately by OpenAI's GPT model using only the information in your files.

---

## ✨ Features

- **Multi-PDF Upload** — Upload one or more PDFs and build a unified knowledge base.
- **Intelligent Chunking** — Documents are split into overlapping chunks for high-quality retrieval.
- **FAISS Vector Search** — Lightning-fast similarity search over your document embeddings.
- **Grounded Answers** — Responses are generated strictly from your documents — no hallucinations.
- **Polished UI** — A clean, dark-themed interface with status indicators and styled answer cards.

---

## 🛠️ Tech Stack

| Layer        | Technology                        |
| ------------ | --------------------------------- |
| UI           | Streamlit                         |
| LLM          | OpenAI `gpt-4o-mini`             |
| Embeddings   | OpenAI `text-embedding-3-small`  |
| Vector Store | FAISS (via `faiss-cpu`)           |
| PDF Parsing  | PyPDF (via LangChain)             |
| Orchestration| LangChain                         |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- An **OpenAI API key** — [get one here](https://platform.openai.com/api-keys)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/converse-pdf.git
   cd converse-pdf
   ```

2. **Create a virtual environment** *(recommended)*

   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   # venv\Scripts\activate    # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

### Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 💡 How to Use

1. Paste your **OpenAI API Key** in the sidebar.
2. Upload one or more **PDF** files.
3. Click **Submit & Process** — the app extracts text, chunks it, and creates vector embeddings.
4. Type a question in the main area and get an AI-generated answer grounded in your documents.

---

## 📁 Project Structure

```
converse-pdf/
├── app.py               # Main Streamlit application (single-file)
├── requirements.txt     # Python dependencies
├── faiss_index/         # Auto-generated FAISS index (after processing)
└── README.md            # You are here
```

---

## ⚙️ Configuration

| Parameter        | Default              | Description                              |
| ---------------- | -------------------- | ---------------------------------------- |
| Chunk size       | 10,000 chars         | Size of each text chunk for embedding    |
| Chunk overlap    | 1,000 chars          | Overlap between consecutive chunks       |
| Embedding model  | `text-embedding-3-small` | OpenAI embedding model             |
| Chat model       | `gpt-4o-mini`        | OpenAI chat model for answer generation  |
| Temperature      | 0.3                  | Controls answer creativity (lower = more precise) |
| Top-K results    | 5                    | Number of chunks retrieved per query     |

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Krishnansh Sharma**

---

> *Built with ❤️ using Streamlit, LangChain, and OpenAI.*
