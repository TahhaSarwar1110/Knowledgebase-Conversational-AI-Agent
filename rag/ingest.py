# rag/ingest.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def get_vector_store(pdf_path: str):
    """
    Load PDF, split into chunks, and create vector store
    """
    try:
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        return vector_store
        
    except Exception as e:
        print(f"Error in get_vector_store: {e}")
        raise

def get_retriever(vector_store):
    """
    Create a retriever from the vector store
    """
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 1, 'score_threshold': 0.7}
        )
        return retriever
    except Exception as e:
        print(f"Error in get_retriever: {e}")
        raise
