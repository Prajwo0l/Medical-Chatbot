from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from src.helper import (
    loadpdf,
    filter_to_minimal_docs,
    add_contextual_headers,
    hierarchical_split,
    download_embedding
)
import os
import time

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"]   = OPENAI_API_KEY

# ── Step 1: Load PDFs ─────────────────────────────────────────────────────────
print("Loading PDFs...")
extracted_data = loadpdf("data")
minimal_docs   = filter_to_minimal_docs(extracted_data)

# ── Step 2: Add contextual headers to every chunk ────────────────────────────
print("Adding contextual headers...")
headed_docs = add_contextual_headers(minimal_docs)

# ── Step 3: Hierarchical chunking ────────────────────────────────────────────
print("Running hierarchical chunking...")
parent_chunks, child_chunks = hierarchical_split(headed_docs)
print(f"  Parent chunks : {len(parent_chunks)}")
print(f"  Child chunks  : {len(child_chunks)}  (these get embedded)")

# ── Step 4: Download embeddings ───────────────────────────────────────────────
print("Loading embedding model (BAAI/bge-base-en-v1.5)...")
embeddings = download_embedding()

# ── Step 5: Connect to Pinecone & auto-create index if missing ───────────────
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

indexes = pc.list_indexes()
print(f"Available indexes: {indexes}")

if index_name in [idx.name for idx in indexes]:
    index = pc.Index(index_name)
    print(f"Connected to existing index: {index_name}")
else:
    print(f"Index '{index_name}' not found — creating it automatically (768 dims, cosine)...")
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    # Wait for the index to be ready
    print("Waiting for index to be ready...")
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(2)
    index = pc.Index(index_name)
    print(f"Index '{index_name}' created successfully.")

# ── Step 6: Index child chunks into Pinecone ──────────────────────────────────
print("Indexing child chunks into Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=child_chunks,
    embedding=embeddings,
    index_name=index_name
)

print(f"Indexing completed. {len(child_chunks)} child chunks stored in Pinecone.")
print("Run python app.py to start the chatbot.")