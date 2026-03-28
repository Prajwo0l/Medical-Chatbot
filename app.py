from flask import Flask, render_template, request, session
from src.helper import download_embedding, rerank_documents
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

embeddings = download_embedding()
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
)

semantic_retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)

chatModel = ChatOpenAI(model="gpt-4o")


multi_query_prompt = PromptTemplate(
    input_variables=["question"],
    template=multi_query_prompt_template
)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=semantic_retriever,
    llm=chatModel,
    prompt=multi_query_prompt
)


print("Loading documents into BM25 index...")
_seed_docs = docsearch.similarity_search("medical", k=200)

bm25_retriever = BM25Retriever.from_documents(_seed_docs)
bm25_retriever.k = 10

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, multi_query_retriever],
    weights=[0.4, 0.6]
    # 60% weight to semantic/multi-query (better for meaning-based medical questions)
    # 40% weight to BM25 (catches exact drug names, dosages, medical codes)
)


compressor = LLMChainExtractor.from_llm(chatModel)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=hybrid_retriever
)


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


    response = rag_chain.invoke({
        "input": msg,
        "chat_history": chat_history
    })


    retrieved_docs = response.get("context", [])
    if retrieved_docs:
        reranked_docs = rerank_documents(msg, retrieved_docs, top_n=3)
    else:
        reranked_docs = []

    answer = response["answer"]

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

    # Save updated history (keep last 10 messages to avoid token overflow)
    raw_history.append({"role": "human", "content": msg})
    raw_history.append({"role": "ai",    "content": answer})
    session["chat_history"] = raw_history[-10:]

    print(f"Question : {msg}")
    print(f"Answer   : {answer}")

    return str(answer)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
