<div align="center">

# 🏥 Medical Chatbot

### Advanced RAG-Powered Medical Q&A with Retrieval-Augmented Generation

An intelligent, document-grounded medical assistant that retrieves context from trusted medical literature and generates concise, accurate answers — built on an **Advanced RAG pipeline** with hybrid search, reranking, query expansion, contextual compression, and conversation memory.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white)](https://platform.openai.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-VectorDB-00B9A5?style=for-the-badge)](https://www.pinecone.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![AWS ECR](https://img.shields.io/badge/AWS-ECR%20%2B%20EC2-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)](https://aws.amazon.com/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/features/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)

</div>

---

> ⚠️ **Medical Disclaimer**
> This chatbot is for **informational and educational purposes only**. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for any medical concerns.

---

## 📖 Overview

The **Medical Chatbot** is a production-ready conversational AI application built on an **Advanced RAG (Retrieval-Augmented Generation)** architecture. It goes far beyond basic RAG by layering multiple retrieval quality improvements on top of each other:

1. **Ingests** medical PDFs into a Pinecone vector database with optimized chunking
2. **Expands** every user question into 3 alternative phrasings (MultiQueryRetriever)
3. **Retrieves** using both semantic search and keyword search combined (Hybrid Search)
4. **Compresses** retrieved chunks to keep only relevant sentences (Contextual Compression)
5. **Reranks** the compressed results using a local cross-encoder model (Free Reranking)
6. **Generates** a grounded answer via GPT-4o with full conversation memory
7. **Cites** the source document and page for every answer

This pipeline dramatically reduces hallucinations and returns highly accurate, relevant, and trustworthy medical answers.

---

## ✨ Features

- 🤖 **Advanced RAG Pipeline** — 4-stage retrieval pipeline for maximum accuracy
- 🔀 **Hybrid Search** — combines semantic (Pinecone) + keyword (BM25) retrieval
- 🔁 **Multi-Query Retrieval** — generates 3 question variants to improve recall
- 🗜️ **Contextual Compression** — strips irrelevant sentences from retrieved chunks
- 🏆 **Cross-Encoder Reranking** — re-scores candidates with a local model (no API cost)
- 💬 **Conversation Memory** — handles follow-up questions using full chat history
- 📄 **Source Citations** — every answer shows the source document and page number
- 🔍 **Optimized Chunking** — chunk size 1200 chars / overlap 150 for full medical sentences
- 🐳 **Fully Dockerized** — one-command containerized deployment
- 🔄 **CI/CD Pipeline** — automated build → ECR → EC2 via GitHub Actions
- 🔐 **Secure Config** — all secrets managed via `.env` and GitHub Secrets

---

## 🧠 Advanced RAG Architecture

```
User Question
      │
      ▼
 Flask Web App (app.py)
      │
      ▼
 ┌─────────────────────────────────────┐
 │  STAGE 1 — Query Expansion          │
 │  MultiQueryRetriever                │
 │  Generates 3 alternative phrasings  │
 │  of the user question               │
 └──────────────┬──────────────────────┘
                │ 3 query variants
                ▼
 ┌─────────────────────────────────────┐
 │  STAGE 2 — Hybrid Search            │
 │  EnsembleRetriever                  │
 │  Semantic (Pinecone) 60% weight     │
 │  Keyword  (BM25)     40% weight     │
 │  Merged via Reciprocal Rank Fusion  │
 └──────────────┬──────────────────────┘
                │ top 10 candidates
                ▼
 ┌─────────────────────────────────────┐
 │  STAGE 3 — Contextual Compression   │
 │  LLMChainExtractor                  │
 │  Keeps only sentences relevant      │
 │  to the question — removes noise    │
 └──────────────┬──────────────────────┘
                │ compressed chunks
                ▼
 ┌─────────────────────────────────────┐
 │  STAGE 4 — Reranking (free, local)  │
 │  cross-encoder/ms-marco-MiniLM-L6   │
 │  Re-scores all candidates           │
 │  Returns top 3 most relevant        │
 └──────────────┬──────────────────────┘
                │ top 3 reranked chunks
                ▼
 GPT-4o — generates answer + cites source
                │
                ▼
 Chat UI — answer displayed with source
```

**Indexing Pipeline (run once via `store_index.py`):**
```
data/*.pdf
    │
    ▼
PyPDFLoader  →  RecursiveCharacterTextSplitter (1200 chars, 150 overlap)
    │
    ▼
HuggingFace Embeddings (all-MiniLM-L6-v2)
    │
    ▼
Pinecone VectorStore  ("medical-chatbot" index)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.10, Flask 3.1 |
| **AI / LLM** | OpenAI GPT-4o via `langchain-openai` |
| **RAG Framework** | LangChain 0.3 |
| **Embeddings** | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Database** | Pinecone |
| **Keyword Search** | BM25 via `rank_bm25` |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, free) |
| **Query Expansion** | LangChain `MultiQueryRetriever` |
| **Compression** | LangChain `ContextualCompressionRetriever` |
| **PDF Loading** | LangChain `PyPDFLoader` + `DirectoryLoader` |
| **Frontend** | HTML, CSS, JavaScript (Jinja2 templates) |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |
| **Cloud** | AWS ECR (image registry) + AWS EC2 (runtime) |

---

## 📋 Prerequisites

- Python **3.10** or higher
- A **Pinecone** account and API key → [pinecone.io](https://www.pinecone.io/)
- An **OpenAI** API key → [platform.openai.com](https://platform.openai.com/)
- **Docker** (optional, for containerized deployment)
- **AWS CLI** configured (optional, for cloud deployment)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Prajwo0l/Medical-Chatbot.git
cd Medical-Chatbot
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv env

# macOS / Linux
source env/bin/activate

# Windows
env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> The cross-encoder reranker model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) downloads automatically from HuggingFace on first startup (~80MB, one time only).

### 4. Configure Environment Variables

Create a `.env` file in the project root and add your API keys:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

> ⚠️ **Never commit your `.env` file.** It is already listed in `.gitignore`.

### 5. Add Your Medical Documents

Place your medical PDF files inside the `data/` directory:

```
data/
└── Medical_book.pdf      ← already included
└── your_document.pdf     ← add more PDFs here
```

### 6. Build the Pinecone Vector Index

Run this once (or whenever you add new documents):

```bash
python store_index.py
```

> 💡 Create a Pinecone index named `medical-chatbot` with **384 dimensions** for `all-MiniLM-L6-v2` before running this script.

### 7. Run the Application

```bash
python app.py
```

Visit `http://localhost:8080` in your browser.

---

## 🐳 Docker Deployment

```bash
docker build -t medical-chatbot:latest .

docker run -d \
  -p 8080:8080 \
  -e PINECONE_API_KEY=your_pinecone_api_key \
  -e OPENAI_API_KEY=your_openai_api_key \
  --name medical-chatbot \
  medical-chatbot:latest
```

---

## ☁️ AWS CI/CD Deployment

Every push to `main` triggers a fully automated GitHub Actions pipeline:

```
Continuous-Integration
  └── Checkout code
  └── Configure AWS credentials
  └── Login to Amazon ECR
  └── Build Docker image
  └── Push image to ECR

Continuous-Deployment
  └── SSH into EC2
  └── Pull latest image from ECR
  └── Run container on port 8080
```

### Required GitHub Secrets

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | AWS IAM access key |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM secret key |
| `AWS_DEFAULT_REGION` | e.g. `us-east-1` |
| `ECR_REPO` | ECR repository name |
| `PINECONE_API_KEY` | Pinecone API key |
| `OPENAI_API_KEY` | OpenAI API key |

---

## 📁 Project Structure

```
Medical-Chatbot/
├── .github/
│   └── workflows/
│       └── cicd.yaml          # GitHub Actions CI/CD pipeline
├── data/
│   └── Medical_book.pdf       # Source medical document(s)
├── research/                  # Jupyter notebooks for experimentation
├── src/
│   ├── __init__.py
│   ├── helper.py              # PDF loading, chunking, embedding, reranker
│   └── prompt.py              # System prompts + multi-query prompt template
├── static/                    # CSS, JS, and static assets
├── templates/
│   └── chat.html              # Chat web interface (Jinja2)
├── .env                       # Environment variables (not committed)
├── .gitignore
├── app.py                     # Flask app + full Advanced RAG pipeline
├── Dockerfile                 # Container build instructions
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── store_index.py             # One-time script to build Pinecone index
└── README.md
```

---

## ⚙️ Configuration Reference

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `PINECONE_API_KEY` | ✅ | API key from Pinecone dashboard |
| `OPENAI_API_KEY` | ✅ | API key from OpenAI platform |

### Key Parameters

| Parameter | Location | Value | Description |
|---|---|---|---|
| `chunk_size` | `helper.py` | `1200` | Characters per text chunk (was 500) |
| `chunk_overlap` | `helper.py` | `150` | Overlap between chunks (was 20) |
| `semantic_k` | `app.py` | `10` | Candidates retrieved from Pinecone |
| `bm25_k` | `app.py` | `10` | Candidates from BM25 keyword search |
| `rerank_top_n` | `app.py` | `3` | Final chunks passed to GPT-4o after reranking |
| `bm25_weight` | `app.py` | `0.4` | Weight of keyword search in hybrid retrieval |
| `semantic_weight` | `app.py` | `0.6` | Weight of semantic search in hybrid retrieval |
| `model` | `app.py` | `gpt-4o` | OpenAI model for generation |
| `reranker_model` | `helper.py` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local reranker |
| `embedding_model` | `helper.py` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `history_limit` | `app.py` | `10` | Max messages kept in session memory |
| `port` | `app.py` | `8080` | Flask server port |

---

## 💬 How It Works — Step by Step

1. User types a medical question in the chat interface
2. Flask receives the message via `POST /get` and loads chat history from session
3. The history-aware retriever reformulates the question if it is a follow-up
4. `MultiQueryRetriever` generates 3 alternative phrasings of the question
5. `EnsembleRetriever` searches Pinecone (semantic) and BM25 (keyword) with all 3 variants, merges results via Reciprocal Rank Fusion
6. `ContextualCompressionRetriever` compresses each chunk — keeping only the sentences relevant to the question
7. `rerank_documents()` scores all compressed chunks with a local cross-encoder and returns the top 3
8. GPT-4o generates a response in 3 sentences or less, grounded in the reranked context
9. Source document name and page number are extracted and appended to the answer
10. Answer is saved to session memory and returned to the browser

---

## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 👤 Author

**Prajwol Lamichhane**
- Email: prajwol.lamichhane@gmail.com
- GitHub: [@Prajwo0l](https://github.com/Prajwo0l)

---

## 📄 License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

---

<div align="center">

*Built with LangChain · Powered by GPT-4o · Deployed on AWS*

</div>
