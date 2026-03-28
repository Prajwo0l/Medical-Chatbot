from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List


def loadpdf(data):
    loader = DirectoryLoader(
        data,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs = []
    for doc in docs:
        minimal_doc = Document(
            page_content=doc.page_content,
            metadata={
                "source": doc.metadata.get("source", ""),
                "page_number": doc.metadata.get("page_number", "")
            }
        )
        minimal_docs.append(minimal_doc)
    return minimal_docs


# chunk_size raised 500 -> 1200, overlap raised 20 -> 150
# Medical sentences were being cut in half at 500 chars. Larger chunks
# preserve full sentences and more context per retrieval.
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150
    )
    texts_chunks = text_splitter.split_documents(minimal_docs)
    return texts_chunks


def download_embedding():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings


# ADVANCED RAG — FREE RERANKER
# Uses a local cross-encoder model (no API key, runs on CPU).
# Takes a list of (query, document) pairs and scores each one.
# Higher score = more relevant to the question.
# Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (small, fast, free on HuggingFace)
def get_reranker():
    from sentence_transformers import CrossEncoder
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return model


def rerank_documents(query: str, docs: List[Document], top_n: int = 3) -> List[Document]:
    """
    Re-scores a list of retrieved documents against the query using a cross-encoder.
    Returns only the top_n most relevant documents.

    Why this is better than just using top-k from Pinecone:
    Pinecone uses fast approximate similarity (cosine distance on embeddings).
    A cross-encoder reads the query AND the document together — much more accurate
    but slower, so we only use it to re-score a small set of candidates.
    """
    reranker = get_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_n]]
