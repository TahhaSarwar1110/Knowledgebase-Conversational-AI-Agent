import os
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter

def get_vector_store(pdf_path: "docs/User Query_2.pdf", persist_directory: str = "./TechnoSurge"):
    """Load PDF, split into chunks, and create/load vector store"""
    
    # Load PDF document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(
        separator=".", 
        chunk_size=500, 
        chunk_overlap=50
    )
    splitted_documents = text_splitter.split_documents(documents)
    
    # Clean up text
    for doc in splitted_documents:
        doc.page_content = ' '.join(doc.page_content.split())
    
    # Create embeddings and vector store
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = Chroma.from_documents(
        embedding=embedding,
        documents=splitted_documents,
        persist_directory=persist_directory
    )
    
    return vector_store

def get_retriever(vector_store):
    """Create retriever from vector store"""
    return vector_store.as_retriever(
        search_type='similarity_score_threshold', 
        search_kwargs={'k': 1, 'score_threshold': 0.8}
    )
