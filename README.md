<div align="center">

# 🏥 Medical Chatbot

### AI-Powered Medical Q&A with Retrieval-Augmented Generation (RAG)

An intelligent, document-grounded medical assistant that retrieves context from trusted medical literature and generates concise, accurate answers — reducing hallucinations common in pure LLM systems.

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

The **Medical Chatbot** is a production-ready conversational AI application built on a **RAG (Retrieval-Augmented Generation)** architecture. Instead of relying solely on an LLM's internal knowledge, it:

1. **Ingests** medical PDFs and documents into a vector database (Pinecone)
2. **Retrieves** the most semantically relevant chunks for each user question
3. **Generates** a concise, grounded answer using GPT-4o — backed by real source material

This approach drastically reduces hallucinations and ensures answers are rooted in your curated medical knowledge base.

---

## ✨ Features

- 🤖 **RAG Pipeline** — document retrieval + LLM generation for grounded, accurate responses
- 📄 **PDF Ingestion** — automatically loads and indexes medical PDFs from the `data/` directory
- 🔍 **Semantic Search** — similarity-based retrieval using HuggingFace sentence embeddings (`all-MiniLM-L6-v2`)
- 💬 **Web Chat Interface** — clean, responsive UI built with Flask + Jinja2 templates
- 🐳 **Fully Dockerized** — one-command containerized deployment
- 🔄 **CI/CD Pipeline** — automated build, push to AWS ECR, and deploy to EC2 via GitHub Actions
- 🔐 **Secure Config** — all secrets managed via `.env` and GitHub Secrets

---

## 🧠 Architecture

```
User Question
      │
      ▼
 Flask Web App (app.py)
      │
      ▼
 LangChain Retrieval Chain
      │
      ├──► Pinecone VectorStore  ──► Top-3 Relevant Chunks
      │         (semantic search)
      │
      ▼
 GPT-4o (ChatOpenAI)
      │  (system prompt + retrieved context + user question)
      │
      ▼
 Concise Medical Answer (max 3 sentences)
      │
      ▼
 Rendered in Chat UI
```

**Indexing Pipeline (run once):**
```
data/*.pdf
    │
    ▼
PyPDFLoader  →  RecursiveCharacterTextSplitter (500 chars, 20 overlap)
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
| **PDF Loading** | LangChain `PyPDFLoader` + `DirectoryLoader` |
| **Frontend** | HTML, CSS, JavaScript (Jinja2 templates) |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |
| **Cloud** | AWS ECR (image registry) + AWS EC2 (runtime) |

---

## 📋 Prerequisites

Before getting started, ensure you have the following:

- Python **3.10** or higher
- `pip` package manager
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
# Create virtual environment
python -m venv env

# Activate — macOS / Linux
source env/bin/activate

# Activate — Windows
env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
touch .env   # macOS / Linux
# or manually create .env on Windows
```

Add your API keys to `.env`:

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

This step reads all PDFs, generates embeddings, and uploads them to Pinecone. **Run this once** (or whenever you add new documents):

```bash
python store_index.py
```

Expected output:
```
Available indexes: ['medical-chatbot']
Connected to the index: medical-chatbot
Indexing completed successfully.
```

> 💡 Make sure a Pinecone index named `medical-chatbot` exists in your Pinecone dashboard before running this script. Create it with **1536 dimensions** if using OpenAI embeddings, or **384 dimensions** for `all-MiniLM-L6-v2`.

### 7. Run the Application

```bash
python app.py
```

Open your browser and navigate to:

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

This project includes a fully automated **GitHub Actions** pipeline (`.github/workflows/cicd.yaml`) that:

1. **Builds** the Docker image
2. **Pushes** it to **AWS Elastic Container Registry (ECR)**
3. **Deploys** and runs it on an **AWS EC2** instance

### Required GitHub Secrets

Navigate to your repository → **Settings → Secrets and variables → Actions** and add:

| Secret Name | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | Your AWS IAM access key ID |
| `AWS_SECRET_ACCESS_KEY` | Your AWS IAM secret access key |
| `AWS_DEFAULT_REGION` | AWS region (e.g., `us-east-1`) |
| `ECR_REPO` | Your ECR repository name |
| `PINECONE_API_KEY` | Your Pinecone API key |
| `OPENAI_API_KEY` | Your OpenAI API key |

### Trigger Deployment

Every push to the `main` branch automatically triggers the full CI/CD pipeline:

```bash
git push origin main
```

The pipeline runs two jobs in sequence:

```
Continuous-Integration
  └── Checkout code
  └── Configure AWS credentials
  └── Login to Amazon ECR
  └── Build Docker image
  └── Push image to ECR

Continuous-Deployment
  └── Pull latest image from ECR
  └── Run container on EC2 with environment variables
```

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
│   ├── helper.py              # PDF loading, chunking, embedding logic
│   └── prompt.py              # System prompt for the LLM
├── static/                    # CSS, JS, and static assets
├── templates/
│   └── chat.html              # Chat web interface (Jinja2)
├── .env                       # Environment variables (not committed)
├── .gitignore
├── app.py                     # Flask application entry point
├── Dockerfile                 # Container build instructions
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup (author: Prajwol Lamichhane)
├── store_index.py             # One-time script to build Pinecone index
└── README.md
```

---

## ⚙️ Configuration Reference

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `PINECONE_API_KEY` | ✅ | API key from your Pinecone dashboard |
| `OPENAI_API_KEY` | ✅ | API key from OpenAI platform |

### Key Parameters

| Parameter | Location | Default | Description |
|---|---|---|---|
| `chunk_size` | `helper.py` | `500` | Characters per text chunk |
| `chunk_overlap` | `helper.py` | `20` | Overlap between chunks |
| `search_k` | `app.py` | `3` | Number of chunks retrieved per query |
| `model` | `app.py` | `gpt-4o` | OpenAI model used for generation |
| `embedding_model` | `helper.py` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `index_name` | `app.py` / `store_index.py` | `medical-chatbot` | Pinecone index name |
| `port` | `app.py` | `8080` | Flask server port |

---

## 💬 How It Works — Step by Step

1. A user types a medical question in the chat interface
2. The Flask app receives the message via `POST /get`
3. LangChain encodes the question using HuggingFace embeddings
4. Pinecone retrieves the **top 3 most relevant document chunks** via cosine similarity
5. The retrieved chunks are injected into the system prompt as `{context}`
6. GPT-4o generates a response in **3 sentences or less**, grounded in the retrieved context
7. The answer is returned to the browser and displayed in the chat UI

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
