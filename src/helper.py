from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

def loadpdf(data):
    loader=DirectoryLoader(data
                           , glob="**/*.pdf",
                           loader_cls=PyPDFLoader)
    documents=loader.load()
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

def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)

    texts_chunks= text_splitter.split_documents(minimal_docs)
    return texts_chunks



def download_embedding():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

embeddings = download_embedding()



