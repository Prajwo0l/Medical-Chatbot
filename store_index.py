from langchain_pinecone import PineconeVectorStore
import os
from src.helper import loadpdf, filter_to_minimal_docs, text_split, download_embedding
from pinecone import Pinecone  # New import for Pinecone class
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Process the documents
extracted_data = loadpdf("data")
minimal_docs = filter_to_minimal_docs(extracted_data)
texts_chunks = text_split(minimal_docs)

# Download embeddings
embeddings = download_embedding()

# Initialize Pinecone with the API key (new approach)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define your index name
index_name = "medical-chatbot"  # Ensure this is the correct index name

# Print available indexes to check if the index exists
indexes = pc.list_indexes()
print("Available indexes:", indexes)

# Ensure the index exists and connect to it
if index_name in indexes:
    # Correct way to create the index object
    index = pc.Index(index_name)  # Now using 'pc.Index()' instead of the previous method
    print(f"Connected to the index: {index_name}")
else:
    print(f"Index '{index_name}' does not exist!")
    exit()

# Use the Pinecone index to create a PineconeVectorStore
docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunks,
    embedding=embeddings,
    index=index  # Ensure the correct Pinecone index object is passed
)

print("Indexing completed successfully.")