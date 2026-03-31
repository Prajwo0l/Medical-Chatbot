from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Tuple
import os


def loadpdf(data):
    loader = DirectoryLoader(
        data,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


# ── MISSING TECHNIQUE 1: Contextual Chunk Headers ─────────────────────────────
# Problem: Chunks had zero context about where they came from. When retrieved,
# GPT-4o had no idea if a chunk was from Chapter 1 or Chapter 20.
# Fix: Prepend every chunk with a header containing the source file and page.
# Example header: "Source: Medical_book.pdf | Page: 42\n\n[chunk content]"
# This helps the model understand the origin and section of every piece of text.
def add_contextual_headers(docs: List[Document]) -> List[Document]:
    headed_docs = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        filename = os.path.basename(source)
        page = doc.metadata.get("page", doc.metadata.get("page_number", "?"))
        header = f"Source: {filename} | Page: {page}\n\n"
        headed_doc = Document(
            page_content=header + doc.page_content,
            metadata=doc.metadata
        )
        headed_docs.append(headed_doc)
    return headed_docs


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs = []
    for doc in docs:
        minimal_doc = Document(
            page_content=doc.page_content,
            metadata={
                "source": doc.metadata.get("source", ""),
                "page_number": doc.metadata.get("page", doc.metadata.get("page_number", ""))
            }
        )
        minimal_docs.append(minimal_doc)
    return minimal_docs


# ── MISSING TECHNIQUE 3: Hierarchical Chunking ────────────────────────────────
# Problem: One chunk size cannot be both precise (for retrieval) and
# context-rich (for answering). Small chunks find the right spot, big
# chunks give enough context to generate a good answer.
# Fix: Create TWO levels of chunks from the same document:
#   - Child chunks (400 chars): small, precise — used for retrieval/embedding
#   - Parent chunks (1200 chars): large, context-rich — sent to GPT-4o
# We store parent_id in each child's metadata so we can look up the parent
# when a child chunk is retrieved.
def hierarchical_split(docs: List[Document]) -> Tuple[List[Document], List[Document]]:
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    parent_chunks = parent_splitter.split_documents(docs)
    child_chunks = []

    for parent_id, parent in enumerate(parent_chunks):
        children = child_splitter.split_documents([parent])
        for child in children:
            child.metadata["parent_id"] = parent_id
            child.metadata["parent_content"] = parent.page_content
            child_chunks.append(child)

    return parent_chunks, child_chunks


# ── MISSING TECHNIQUE 4: Sentence-Window Retrieval ───────────────────────────
# Problem: Embedding at the full chunk level is too coarse — the embedding
# represents the average meaning of 1200 characters, so precise sentences
# get diluted.
# Fix: Split into individual sentences for fine-grained embedding.
# When a sentence is retrieved, expand it to include ±3 surrounding sentences
# so GPT-4o still gets enough context to generate a complete answer.
def sentence_window_split(docs: List[Document], window: int = 3) -> List[Document]:
    sentence_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0,
        separators=[". ", "! ", "? ", "\n"]
    )
    sentences = sentence_splitter.split_documents(docs)

    windowed = []
    for i, sent in enumerate(sentences):
        start = max(0, i - window)
        end   = min(len(sentences), i + window + 1)
        context_sentences = [s.page_content for s in sentences[start:end]]
        windowed_content = " ".join(context_sentences)

        windowed.append(Document(
            page_content=windowed_content,
            metadata={
                **sent.metadata,
                "core_sentence": sent.page_content,
                "window_size": window
            }
        ))
    return windowed


# ── Standard chunk split (used as default in store_index.py) ─────────────────
# chunk_size raised 500 -> 1200, overlap raised 20 -> 150
def text_split(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)


# ── MISSING TECHNIQUE 5: Better Embeddings ───────────────────────────────────
# Problem: all-MiniLM-L6-v2 is a general-purpose model (384 dimensions).
# It was not trained on medical or scientific text specifically.
# Fix: Upgrade to BAAI/bge-base-en-v1.5 — free on HuggingFace, same size,
# consistently ranks higher on retrieval benchmarks including medical QA tasks.
# No code changes needed in Pinecone — just swap the model name here.
# IMPORTANT: If you already have an existing Pinecone index built with
# all-MiniLM-L6-v2, you must re-run store_index.py after this change
# because the vector dimensions will be different (768 vs 384).
def download_embedding():
    model_name = "BAAI/bge-base-en-v1.5"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embeddings


# ── Free local cross-encoder reranker ────────────────────────────────────────
# Already implemented — kept here unchanged.
# Uses cross-encoder/ms-marco-MiniLM-L-6-v2 (free, local, no API needed).
def get_reranker():
    from sentence_transformers import CrossEncoder
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return model


def rerank_documents(query: str, docs: List[Document], top_n: int = 3) -> List[Document]:
    """
    Re-scores retrieved documents against the query using a local cross-encoder.
    Returns only the top_n most relevant. No API key needed — runs on CPU.
    """
    if not docs:
        return []
    reranker = get_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_n]]