from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
from src.helper import (
    loadpdf,
    filter_to_minimal_docs,
    add_contextual_headers,
    hierarchical_split,
    download_embedding
)
import os

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
# TECHNIQUE 1 — Contextual Chunk Headers
# Each chunk gets a header: "Source: Medical_book.pdf | Page: 42"
# This tells GPT-4o exactly where the chunk came from during generation.
print("Adding contextual headers...")
headed_docs = add_contextual_headers(minimal_docs)

# ── Step 3: Hierarchical chunking ────────────────────────────────────────────
# TECHNIQUE 3 — Hierarchical Chunking
# Creates two levels:
#   parent_chunks (1200 chars) — large, context-rich, stored for reference
#   child_chunks  (400 chars)  — small, precise, embedded into Pinecone
# Each child chunk stores its parent content in metadata so the full context
# can be retrieved even though we embed and search on the smaller child.
print("Running hierarchical chunking...")
parent_chunks, child_chunks = hierarchical_split(headed_docs)
print(f"  Parent chunks : {len(parent_chunks)}")
print(f"  Child chunks  : {len(child_chunks)}  (these get embedded)")

# ── Step 4: Download embeddings ───────────────────────────────────────────────
# TECHNIQUE 5 — Better embeddings
# Using BAAI/bge-base-en-v1.5 (768-dim) instead of all-MiniLM-L6-v2 (384-dim)
# IMPORTANT: Make sure your Pinecone index is created with 768 dimensions.
print("Loading embedding model (BAAI/bge-base-en-v1.5)...")
embeddings = download_embedding()

# ── Step 5: Connect to Pinecone ───────────────────────────────────────────────
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

indexes = pc.list_indexes()
print(f"Available indexes: {indexes}")

if index_name in [idx.name for idx in indexes]:
    index = pc.Index(index_name)
    print(f"Connected to index: {index_name}")
else:
    print(f"Index '{index_name}' does not exist!")
    print("Create it in your Pinecone dashboard with 768 dimensions (for bge-base-en-v1.5).")
    exit()

# ── Step 6: Index child chunks into Pinecone ──────────────────────────────────
# We embed and store the child chunks (400 chars) for precise retrieval.
# Each child's metadata contains parent_content so we always have the full
# context available when the child is retrieved during Q&A.
print("Indexing child chunks into Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=child_chunks,
    embedding=embeddings,
    index_name=index_name
)

print(f"Indexing completed. {len(child_chunks)} child chunks stored in Pinecone.")
print("Run python app.py to start the chatbot.")
