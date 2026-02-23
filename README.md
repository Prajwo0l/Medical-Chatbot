# Medical-Chatbot

## ECR Repo
url : 460628079661.dkr.ecr.ap-southeast-2.amazonaws.com/medicalbot<div align="center">

# Medical Chatbot

**AI-powered medical consultation assistant using Retrieval-Augmented Generation (RAG)**  
An intelligent chatbot that retrieves reliable medical information from documents and provides helpful, context-aware responses.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Container-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-ECR-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)](https://aws.amazon.com/ecr/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/Prajwo0l/Medical-Chatbot/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)

</div>

## 📖 About The Project

This project is a **Medical Chatbot** built to assist users with health-related questions by combining document retrieval with generative AI (RAG architecture).  

It indexes medical documents / PDFs / datasets → retrieves relevant context → generates accurate and grounded answers → reduces hallucinations compared to pure LLM usage.

Main goals:
- Provide helpful, evidence-based medical information
- Offer an easy-to-use web interface
- Support local development and cloud deployment
- Demonstrate modern MLOps practices (Docker + GitHub Actions + AWS ECR)

**⚠️ Important disclaimer**  
This is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.

## ✨ Features

- Conversational interface for symptom checking, condition explanation, treatment info, etc.
- Retrieval-Augmented Generation (RAG) pipeline
- Document indexing & vector store creation
- Responsive web UI (HTML + CSS + JS)
- Fully containerized with Docker
- Automated CI/CD: build → test → push to AWS ECR on every push to main
- Clean project structure for research + production code

## 🛠️ Tech Stack

| Layer              | Technology                          |
|--------------------|-------------------------------------|
| Backend            | Python (Flask / FastAPI style)      |
| Web Framework      | Flask (templates + static files)    |
| AI / RAG           | LangChain / LlamaIndex style + embeddings |
| Vector Store       | FAISS / Chroma / Pinecone (via `store_index.py`) |
| Frontend           | HTML, CSS, JavaScript (Jinja2)      |
| Containerization   | Docker                              |
| CI/CD              | GitHub Actions                      |
| Cloud Registry     | AWS Elastic Container Registry (ECR)|
| Data / Notebooks   | Jupyter notebooks in `/research`    |

## 🚀 Quick Start

### Option 1 – Docker (recommended)

```bash
# Clone the repository
git clone https://github.com/Prajwo0l/Medical-Chatbot.git
cd Medical-Chatbot

# Build the Docker image
docker build -t medical-chatbot:latest .

# Run the container
docker run -d -p 5000:5000 --name medical-chatbot medical-chatbot:latest