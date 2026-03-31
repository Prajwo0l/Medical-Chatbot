# =============================================================================
# backend.py  —  Medical Chatbot with LangGraph Thread Persistence
# Inspired by LangGraph-Core-Components/Chatbot/langraph_backend.py
# =============================================================================

import os
import sqlite3
import uuid
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, MultiQueryRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

from src.helper import download_embedding, rerank_documents, sentence_window_split
from src.prompt import conversational_system_prompt, multi_query_prompt_template

load_dotenv()
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY", "")
os.environ["OPENAI_API_KEY"]   = os.getenv("OPENAI_API_KEY", "")


# =============================================================================
# SQLite — thread persistence
# Stores: thread_id, title, created_at, updated_at
# Each thread maps to a list of messages stored in a separate messages table.
# This mirrors how SqliteSaver works in the LangGraph Chatbot project.
# =============================================================================

DB_PATH = "chatbot_threads.db"

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS threads (
                thread_id  TEXT PRIMARY KEY,
                title      TEXT NOT NULL DEFAULT 'New Chat',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id  TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (thread_id) REFERENCES threads(thread_id)
            )
        """)
        conn.commit()


# ── Thread CRUD ───────────────────────────────────────────────────────────────

def create_thread(title: str = "New Chat") -> str:
    """Create a new thread and return its ID."""
    thread_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO threads (thread_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (thread_id, title, now, now)
        )
        conn.commit()
    return thread_id


def get_all_threads() -> list[dict]:
    """Return all threads ordered by most recently updated."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT thread_id, title, created_at, updated_at FROM threads ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_thread(thread_id: str) -> Optional[dict]:
    """Return a single thread by ID."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT thread_id, title, created_at, updated_at FROM threads WHERE thread_id = ?",
            (thread_id,)
        ).fetchone()
    return dict(row) if row else None


def update_thread_title(thread_id: str, title: str) -> None:
    """Update thread title and updated_at timestamp."""
    now = datetime.now().isoformat()
    with _get_conn() as conn:
        conn.execute(
            "UPDATE threads SET title = ?, updated_at = ? WHERE thread_id = ?",
            (title, now, thread_id)
        )
        conn.commit()


def delete_thread(thread_id: str) -> None:
    """Delete a thread and all its messages."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM threads WHERE thread_id = ?",  (thread_id,))
        conn.commit()


# ── Message CRUD ──────────────────────────────────────────────────────────────

def save_message(thread_id: str, role: str, content: str) -> None:
    """Persist a message and bump the thread's updated_at."""
    now = datetime.now().isoformat()
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (thread_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (thread_id, role, content, now)
        )
        conn.execute(
            "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
            (now, thread_id)
        )
        conn.commit()


def get_messages(thread_id: str) -> list[dict]:
    """Return all messages for a thread in chronological order."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages WHERE thread_id = ? ORDER BY id ASC",
            (thread_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_chat_history(thread_id: str) -> List[BaseMessage]:
    """Return LangChain message objects for the thread (used by the RAG chain)."""
    msgs = get_messages(thread_id)
    history: List[BaseMessage] = []
    for m in msgs:
        if m["role"] == "human":
            history.append(HumanMessage(content=m["content"]))
        elif m["role"] == "ai":
            history.append(AIMessage(content=m["content"]))
    return history


# =============================================================================
# RAG pipeline (same advanced pipeline from app.py)
# Extracted here so backend.py owns everything and app.py stays thin.
# =============================================================================

def build_rag_chain():
    """Build and return the full advanced RAG chain."""
    embeddings     = download_embedding()
    index_name     = "medical-chatbot"

    docsearch = PineconeVectorStore.from_existing_index(
        embedding=embeddings,
        index_name=index_name
    )

    semantic_retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )

    chat_model = ChatOpenAI(model="gpt-4o")

    # Multi-query expansion
    multi_query_prompt = PromptTemplate(
        input_variables=["question"],
        template=multi_query_prompt_template
    )
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=semantic_retriever,
        llm=chat_model,
        prompt=multi_query_prompt
    )

    # Sentence-window BM25
    print("Building sentence-window BM25 index...")
    seed_docs     = docsearch.similarity_search("medical", k=200)
    windowed_docs = sentence_window_split(seed_docs, window=3)
    bm25_retriever = BM25Retriever.from_documents(windowed_docs)
    bm25_retriever.k = 10

    # Hybrid search
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, multi_query_retriever],
        weights=[0.4, 0.6]
    )

    # Contextual compression
    compressor = LLMChainExtractor.from_llm(chat_model)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=hybrid_retriever
    )

    # History-aware retriever
    history_aware_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human",
         "Given the above conversation, generate a standalone search query "
         "that captures what the user is asking. Return only the query, nothing else.")
    ])
    history_aware_ret = create_history_aware_retriever(
        chat_model,
        compression_retriever,
        history_aware_prompt
    )

    # Final QA chain
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", conversational_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    question_answering_chain = create_stuff_documents_chain(chat_model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_ret, question_answering_chain)

    return rag_chain, docsearch


def answer_question(
    rag_chain,
    thread_id: str,
    question: str
) -> str:
    """
    Run the full RAG pipeline for a given thread.
    Loads history from SQLite, invokes chain, saves messages, returns answer.
    """
    # 1. Load this thread's history from DB
    chat_history = get_chat_history(thread_id)

    # 2. Run the chain
    response = rag_chain.invoke({
        "input": question,
        "chat_history": chat_history
    })

    # 3. Rerank retrieved docs
    retrieved_docs = response.get("context", [])
    reranked_docs  = rerank_documents(question, retrieved_docs, top_n=3) if retrieved_docs else []

    answer = response["answer"]

    # 4. Append source citations
    if reranked_docs:
        sources = set()
        for doc in reranked_docs:
            source = doc.metadata.get("source", "")
            page   = doc.metadata.get("page_number", "")
            if source:
                filename = os.path.basename(source)
                label = f"{filename} (page {page})" if page else filename
                sources.add(label)
        if sources:
            answer += f"\n\n📄 Source: {', '.join(sources)}"

    # 5. Persist both messages to SQLite
    save_message(thread_id, "human", question)
    save_message(thread_id, "ai",    answer)

    # 6. Auto-title the thread from the first question
    thread = get_thread(thread_id)
    if thread and thread["title"] == "New Chat":
        update_thread_title(thread_id, question[:45] + ("…" if len(question) > 45 else ""))

    return answer


# Bootstrap DB on import
init_db()