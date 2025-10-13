import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import your existing RAG chain
from chatbot import memory_rag_chain  # <-- uses your same logic

load_dotenv()

# FastAPI instance
app = FastAPI()

# Enable CORS for all domains (adjust later for production security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic request body model
class ChatRequest(BaseModel):
    query: str

@app.get("/")
def root():
    return {"message": "TechnoSurge AI Chatbot API running!"}

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        user_query = request.query
        response = memory_rag_chain.invoke(user_query)
        return {"answer": response}
    except Exception as e:
        return {"error": str(e)}
