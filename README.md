<div align="center">

# 🏥 Medical Chatbot

### Production-Grade Medical Q&A with a 7-Stage Advanced RAG Pipeline

An intelligent, document-grounded medical assistant that retrieves answers from trusted medical literature using a fully implemented advanced RAG architecture — featuring contextual chunk headers, hierarchical chunking, sentence-window retrieval, hybrid search, multi-query expansion, contextual compression, cross-encoder reranking, and conversation memory.

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

The **Medical Chatbot** is a production-ready conversational AI application built on a fully implemented **Advanced RAG (Retrieval-Augmented Generation)** pipeline. It goes well beyond basic RAG by applying seven retrieval quality improvements at both indexing and query time.

Instead of simply embedding raw text and doing a cosine similarity search, this system:

1. Prepends every chunk with source and page metadata before indexing
2. Creates a parent-child chunk hierarchy for precision retrieval with rich context
3. Builds a sentence-window index for fine-grained keyword matching
4. Expands every user question into multiple alternative phrasings
5. Combines semantic and keyword search results via Reciprocal Rank Fusion
6. Compresses retrieved chunks to keep only question-relevant sentences
7. Re-scores all candidates with a local cross-encoder and returns only the top 3

Every answer is grounded in real medical documents, cited with source and page number, and aware of the full conversation history.

---

## ✨ Feature Highlights

| Feature | Implementation |
|---|---|
| Contextual chunk headers | Source filename + page prepended to every chunk at indexing time |
| Hierarchical chunking | Parent chunks (1200 chars) for context, child chunks (400 chars) for precision |
| Sentence-window retrieval | Sentence-level BM25 with ±3 sentence context window |
| Multi-query expansion | 3 alternative phrasings generated per question via GPT-4o |
| Hybrid search | Pinecone semantic (60%) + BM25 keyword (40%) via EnsembleRetriever |
| Contextual compression | LLMChainExtractor strips irrelevant sentences from retrieved chunks |
| Cross-encoder reranking | Free local model re-scores candidates, returns top 3 (no API cost) |
| Conversation memory | History-aware retriever + Flask session — follow-up questions work correctly |
| Source citations | Document name and page number appended to every answer |
| Upgraded embeddings | BAAI/bge-base-en-v1.5 (768-dim) replacing all-MiniLM-L6-v2 (384-dim) |
| CI/CD pipeline | GitHub Actions → Docker build → AWS ECR → AWS EC2 auto-deploy |

---

## 🧠 Architecture

### Indexing Pipeline — run once via `store_index.py`

```
data/*.pdf
    │
    ▼
PyPDFLoader — loads all PDF pages
    │
    ▼
filter_to_minimal_docs() — extracts source + page metadata
    │
    ▼
add_contextual_headers()                    ← TECHNIQUE 1
Prepends "Source: file.pdf | Page: N" to every chunk
    │
    ▼
hierarchical_split()                        ← TECHNIQUE 3
├── Parent chunks (1200 chars, 150 overlap) — stored in metadata
└── Child chunks  (400 chars,  50 overlap)  — embedded into Pinecone
    │
    ▼
BAAI/bge-base-en-v1.5 embeddings (768-dim) ← TECHNIQUE 5
    │
    ▼
Pinecone VectorStore ("medical-chatbot" index)
Child chunks stored. Each child carries parent_content in metadata.
```

### Query Pipeline — runs on every user message in `app.py`

```
User question
    │
    ▼
History-aware reformulation
Converts follow-up questions into standalone queries using chat history
    │
    ▼
MultiQueryRetriever                         ← TECHNIQUE 2 (query expansion)
Generates 3 alternative phrasings → searches Pinecone with all 3
    │
    ▼
EnsembleRetriever (Hybrid Search)           ← TECHNIQUE 6 (hybrid search)
├── Pinecone semantic search (60% weight)
└── BM25 sentence-window keyword search (40% weight)   ← TECHNIQUE 4
    merged via Reciprocal Rank Fusion
    │
    ▼
ContextualCompressionRetriever              ← contextual compression
LLMChainExtractor removes irrelevant sentences from each chunk
    │
    ▼
rerank_documents()                          ← TECHNIQUE 7 (reranking)
cross-encoder/ms-marco-MiniLM-L-6-v2
Re-scores all candidates → returns top 3
    │
    ▼
GPT-4o (gpt-4o)
Generates answer from reranked context + chat history
    │
    ▼
Source citation appended (filename + page)
    │
    ▼
Answer displayed in chat UI + saved to session memory
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Backend** | Python 3.10, Flask 3.1 | Web server and routing |
| **LLM** | OpenAI GPT-4o | Answer generation |
| **RAG Framework** | LangChain 0.3 | Retrieval chains and orchestration |
| **Embeddings** | BAAI/bge-base-en-v1.5 (768-dim) | Semantic vector encoding |
| **Vector Database** | Pinecone | Semantic similarity search |
| **Keyword Search** | BM25 via rank_bm25 | Exact term matching |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 | Local relevance re-scoring |
| **Query Expansion** | LangChain MultiQueryRetriever | Alternative question generation |
| **Compression** | LangChain ContextualCompressionRetriever | Chunk noise reduction |
| **Hybrid Fusion** | LangChain EnsembleRetriever | Reciprocal Rank Fusion |
| **PDF Loading** | PyPDFLoader + DirectoryLoader | Document ingestion |
| **Frontend** | HTML, CSS, JavaScript (Jinja2) | Chat interface |
| **Containerization** | Docker | Portable deployment |
| **CI/CD** | GitHub Actions | Automated build and deploy |
| **Cloud** | AWS ECR + AWS EC2 | Image registry and runtime |

---

## 📋 Prerequisites

- Python **3.10** or higher
- A **Pinecone** account with an index created at **768 dimensions** → [pinecone.io](https://www.pinecone.io/)
- An **OpenAI** API key → [platform.openai.com](https://platform.openai.com/)
- **Docker** (optional, for containerized deployment)
- **AWS CLI** configured (optional, for cloud deployment)

> ⚠️ **Important:** This project uses `BAAI/bge-base-en-v1.5` which produces **768-dimensional** vectors. Your Pinecone index must be created with **768 dimensions**. If you previously used `all-MiniLM-L6-v2` (384-dim), delete the old index and create a new one.

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

> The reranker model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) and embedding model (`BAAI/bge-base-en-v1.5`) download automatically from HuggingFace on first run. Combined download is approximately 500MB and happens only once.

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

> ⚠️ Never commit your `.env` file. It is already listed in `.gitignore`.

### 5. Add Medical Documents

Place your PDF files in the `data/` directory:

```
data/
└── Medical_book.pdf       ← included by default
└── your_document.pdf      ← add more PDFs here
```

### 6. Create Pinecone Index

In your Pinecone dashboard, create an index with these settings:

| Setting | Value |
|---|---|
| Index name | `medical-chatbot` |
| Dimensions | `768` |
| Metric | `cosine` |

### 7. Build the Vector Index

Run this once (or whenever you add new documents). This applies contextual headers, performs hierarchical chunking, and uploads child chunks to Pinecone:

```bash
python store_index.py
```

Expected output:
```
Loading PDFs...
Adding contextual headers...
Running hierarchical chunking...
  Parent chunks : 312
  Child chunks  : 847  (these get embedded)
Loading embedding model (BAAI/bge-base-en-v1.5)...
Connected to index: medical-chatbot
Indexing child chunks into Pinecone...
Indexing completed. 847 child chunks stored in Pinecone.
Run python app.py to start the chatbot.
```

### 8. Run the Application

```bash
python app.py
```

Open your browser and visit:

```
http://localhost:8080
```

---

## 🐳 Docker Deployment

### Build the Image

```bash
docker build -t medical-chatbot:latest .
```

### Run the Container

```bash
docker run -d \
  -p 8080:8080 \
  -e PINECONE_API_KEY=your_pinecone_api_key \
  -e OPENAI_API_KEY=your_openai_api_key \
  --name medical-chatbot \
  medical-chatbot:latest
```

Visit `http://localhost:8080` in your browser.

---

## ☁️ AWS CI/CD Deployment

Every push to the `main` branch triggers a fully automated GitHub Actions pipeline:

```
Continuous-Integration
  └── Checkout code
  └── Configure AWS credentials
  └── Login to Amazon ECR
  └── Build Docker image
  └── Push image to ECR

Continuous-Deployment
  └── SSH into EC2 instance
  └── Pull latest image from ECR
  └── Stop old container
  └── Run new container on port 8080
```

### Required GitHub Secrets

Go to your repository → **Settings → Secrets and variables → Actions** and add:

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | AWS IAM access key ID |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM secret access key |
| `AWS_DEFAULT_REGION` | AWS region (e.g. `us-east-1`) |
| `ECR_REPO` | ECR repository name |
| `PINECONE_API_KEY` | Pinecone API key |
| `OPENAI_API_KEY` | OpenAI API key |

---

## 📁 Project Structure

```
Medical-Chatbot/
├── .github/
│   └── workflows/
│       └── cicd.yaml              # GitHub Actions CI/CD pipeline
├── data/
│   └── Medical_book.pdf           # Source medical document(s)
├── research/                      # Jupyter notebooks for experimentation
├── src/
│   ├── __init__.py
│   ├── helper.py                  # Core RAG utilities:
│   │                              #   loadpdf, filter_to_minimal_docs
│   │                              #   add_contextual_headers (technique 1)
│   │                              #   hierarchical_split     (technique 3)
│   │                              #   sentence_window_split  (technique 4)
│   │                              #   download_embedding     (technique 5)
│   │                              #   rerank_documents       (technique 7)
│   └── prompt.py                  # System prompts + multi-query template
├── static/                        # CSS, JS, static assets
├── templates/
│   └── chat.html                  # Chat web interface (Jinja2)
├── .env                           # API keys — never commit this
├── .gitignore
├── app.py                         # Flask app + full query-time RAG pipeline
├── Dockerfile                     # Container build instructions
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
├── store_index.py                 # Indexing pipeline — run once
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

| Parameter | File | Value | Description |
|---|---|---|---|
| `embedding_model` | `helper.py` | `BAAI/bge-base-en-v1.5` | HuggingFace embedding model (768-dim) |
| `reranker_model` | `helper.py` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local cross-encoder reranker |
| `parent_chunk_size` | `helper.py` | `1200` | Parent chunk size for hierarchical split |
| `parent_chunk_overlap` | `helper.py` | `150` | Parent chunk overlap (12.5%) |
| `child_chunk_size` | `helper.py` | `400` | Child chunk size for precise embedding |
| `child_chunk_overlap` | `helper.py` | `50` | Child chunk overlap |
| `sentence_window` | `helper.py` | `3` | Sentences of context around each retrieved sentence |
| `semantic_k` | `app.py` | `10` | Candidates retrieved from Pinecone |
| `bm25_k` | `app.py` | `10` | Candidates from BM25 keyword search |
| `semantic_weight` | `app.py` | `0.6` | Hybrid search weight for semantic results |
| `bm25_weight` | `app.py` | `0.4` | Hybrid search weight for keyword results |
| `rerank_top_n` | `app.py` | `3` | Final chunks passed to GPT-4o after reranking |
| `history_limit` | `app.py` | `10` | Max messages kept in session memory |
| `model` | `app.py` | `gpt-4o` | OpenAI model for generation |
| `port` | `app.py` | `8080` | Flask server port |

---

## 💬 How It Works — Step by Step

**At indexing time (`store_index.py`):**

1. All PDF files in `data/` are loaded page by page using `PyPDFLoader`
2. Each page is enriched with a contextual header: `Source: filename | Page: N`
3. Documents are split hierarchically into parent chunks (1200 chars) and child chunks (400 chars)
4. Each child chunk stores the full parent content in its metadata for context retrieval
5. Child chunks are embedded using `BAAI/bge-base-en-v1.5` and stored in Pinecone

**At query time (`app.py`):**

1. User types a question in the chat interface
2. Flask loads the chat history from the session
3. The history-aware retriever reformulates follow-up questions into standalone queries
4. `MultiQueryRetriever` generates 3 alternative phrasings of the question
5. `EnsembleRetriever` searches Pinecone (semantic) and a sentence-window BM25 index (keyword) with all variants and merges results via Reciprocal Rank Fusion
6. `ContextualCompressionRetriever` strips sentences from each chunk that are not relevant to the question
7. `rerank_documents()` scores all compressed chunks with a local cross-encoder and returns the top 3
8. GPT-4o generates a concise answer (max 3 sentences) grounded in the reranked context
9. Source document and page number are extracted and appended to the answer
10. The answer and question are saved to session memory and returned to the browser

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
