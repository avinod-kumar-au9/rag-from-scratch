# RAG From Scratch

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from company documents. Built from scratch with Cerebras (LLaMA 3.3 70B), ChromaDB, and Streamlit.

---

## What is in this repo

```
RAG from scratch/
├── RAG_From_scratch.py    # Core RAG logic: document loading, chunking, vector search, LLM agent
├── streamlit_app.py       # Streamlit chat UI
├── hr_document.txt        # Company HR policy document (source data)
├── requirements.txt       # Python dependencies
├── Dockerfile.txt         # Container definition
├── .env                   # Local secrets — never committed (gitignored)
└── .gitignore
```

---

## How it works

```
User types a question
        |
        v
ChromaDB searches hr_document.txt
(finds top 3 most relevant paragraphs)
        |
        v
Those paragraphs are sent to Cerebras LLM as context
(via a tool-call-based RAG agent)
        |
        v
LLM generates a grounded answer
        |
        v
Answer displayed in the Streamlit chat UI
```

Documents are chunked by paragraph at startup and stored in an in-memory ChromaDB collection. No external database needed — everything resets on each restart.

---

## Prerequisites

- Python 3.9+
- A Cerebras API key — https://cloud.cerebras.ai

---

## Local Setup

### Step 1: Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Mac / Linux
# .venv\Scripts\activate         # Windows
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Add your Cerebras API key

Create a `.env` file in the project root:

```
CEREBRAS_API_KEY=csk-xxxxxxxxxxxxxxxxxxxxxxxx
```

---

## Running Locally

### Streamlit UI

```bash
cd "rag-from-scratch"
streamlit run streamlit_app.py
```

Open: http://localhost:8501

---

## Docker

```bash
docker build -f Dockerfile.txt -t rag-app .
docker run -p 9000:9000 --env-file .env rag-app
```

Open: http://localhost:9000

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CEREBRAS_API_KEY is not set` | Check `.env` has `CEREBRAS_API_KEY=csk-...` |
| `hr_document.txt` not found | Run from the `RAG from scratch` folder |
| Streamlit won't start | Make sure streamlit is installed: `pip install streamlit` |
