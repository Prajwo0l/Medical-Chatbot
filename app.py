from flask import Flask, render_template, request, session
from src.helper import download_embedding, rerank_documents, sentence_window_split
from src.prompt import conversational_system_prompt, multi_query_prompt_template
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.retrievers import BM25Retriever, EnsembleRetriever, MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"]  = OPENAI_API_KEY

# ── TECHNIQUE 5: Better embeddings (BAAI/bge-base-en-v1.5) ───────────────────
# Upgraded from all-MiniLM-L6-v2 (general purpose, 384-dim) to
# BAAI/bge-base-en-v1.5 (768-dim, better on retrieval benchmarks).
# NOTE: Re-run store_index.py if you had an existing Pinecone index —
# the vector dimensions changed from 384 to 768.
embeddings = download_embedding()
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
)

# ── Base semantic retriever (Pinecone) ────────────────────────────────────────
# k=10 gives enough candidates for reranking to pick the best 3 from.
semantic_retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)

chatModel = ChatOpenAI(model="gpt-4o")

# ── Multi-Query Retrieval ─────────────────────────────────────────────────────
# Generates 3 alternative phrasings of the question before searching Pinecone.
# Increases recall — different phrasings match different chunks.
multi_query_prompt = PromptTemplate(
    input_variables=["question"],
    template=multi_query_prompt_template
)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=semantic_retriever,
    llm=chatModel,
    prompt=multi_query_prompt
)

# ── TECHNIQUE 4: Sentence-Window Retrieval via BM25 ──────────────────────────
# Problem: Embedding full 1200-char chunks dilutes precise sentence meaning.
# Fix: Load seed documents, split them into sentence-window chunks (each sentence
# surrounded by ±3 neighbouring sentences), and feed those to BM25.
# This gives fine-grained keyword matching while each result still contains
# surrounding context — so GPT-4o gets enough to work with.
print("Building sentence-window BM25 index...")
_seed_docs = docsearch.similarity_search("medical", k=200)
_windowed_docs = sentence_window_split(_seed_docs, window=3)

bm25_retriever = BM25Retriever.from_documents(_windowed_docs)
bm25_retriever.k = 10

# ── Hybrid Search (BM25 + Semantic) ─────────────────────────────────────────
# Merges sentence-window BM25 results (exact keyword matches, fine-grained)
# with multi-query semantic results (meaning-based, broader recall).
# Weights: 40% BM25 / 60% semantic — tuned for medical Q&A.
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, multi_query_retriever],
    weights=[0.4, 0.6]
)

# ── Contextual Compression ───────────────────────────────────────────────────
# After retrieval, strips sentences from each chunk that are NOT relevant
# to the question. Reduces noise before passing context to GPT-4o.
compressor = LLMChainExtractor.from_llm(chatModel)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=hybrid_retriever
)

# ── Conversation memory — history-aware retriever ────────────────────────────
# Reformulates follow-up questions ("what are its symptoms?") into a
# self-contained query ("what are the symptoms of diabetes?") using
# chat history before searching.
history_aware_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("human",
     "Given the above conversation, generate a standalone search query "
     "that captures what the user is asking. Return only the query, nothing else.")
])

history_aware_ret = create_history_aware_retriever(
    chatModel,
    compression_retriever,
    history_aware_prompt
)

# ── Final QA chain ────────────────────────────────────────────────────────────
# TECHNIQUE 1 (Contextual Chunk Headers) is applied at indexing time in
# store_index.py via add_contextual_headers(). By the time chunks arrive
# here, they already contain "Source: file.pdf | Page: N" headers, so
# GPT-4o knows exactly where every piece of context came from.
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", conversational_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

question_answering_chain = create_stuff_documents_chain(chatModel, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_ret, question_answering_chain)


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    session.clear()
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]

    # Rebuild chat history from session
    raw_history = session.get("chat_history", [])
    chat_history = []
    for item in raw_history:
        if item["role"] == "human":
            chat_history.append(HumanMessage(content=item["content"]))
        else:
            chat_history.append(AIMessage(content=item["content"]))

    # Run full pipeline: MultiQuery → Hybrid → Compression → History-aware
    response = rag_chain.invoke({
        "input": msg,
        "chat_history": chat_history
    })

    # Cross-encoder reranking — final quality filter on retrieved context
    # Picks the 3 most relevant chunks from all retrieved candidates
    retrieved_docs = response.get("context", [])
    reranked_docs  = rerank_documents(msg, retrieved_docs, top_n=3) if retrieved_docs else []

    answer = response["answer"]

    # Source citations — extracted from reranked docs
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

    # Keep last 10 messages in session to avoid token overflow
    raw_history.append({"role": "human", "content": msg})
    raw_history.append({"role": "ai",    "content": answer})
    session["chat_history"] = raw_history[-10:]

    print(f"Question : {msg}")
    print(f"Answer   : {answer}")

    return str(answer)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
