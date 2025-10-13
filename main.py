# app/main.py
import os
import time
import uuid
import asyncio
from typing import Dict, Optional, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# RAG helpers
from rag.ingest import get_vector_store, get_retriever
from rag.chain import create_rag_chain

# Config
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", 60 * 60))  # default 1 hour
CLEANUP_INTERVAL_SECONDS = int(os.getenv("CLEANUP_INTERVAL_SECONDS", 60))  # run every minute
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")  # e.g. "https://site.com"

# FastAPI app
app = FastAPI(title="TechnoSurge RAG API", version="1.0.0")

# CORS - in prod set actual origins in ALLOWED_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/response models
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

class IngestResponse(BaseModel):
    message: str
    status: str

# Globals
vector_store: Any = None
retriever: Any = None
# We will create a chain+memory per session and store here
chat_sessions: Dict[str, Dict[str, Any]] = {}  # session_id -> {"chain":..., "memory":..., "last_activity": float}

# Startup: initialize vector store & retriever
@app.on_event("startup")
async def startup_event():
    global vector_store, retriever

    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is missing")

        pdf_path = os.getenv("PDF_PATH", "docs/User Query_2.pdf")
        vector_store = get_vector_store(pdf_path)
        retriever = get_retriever(vector_store)

        # Warm a prototype chain (optional) to avoid first-call cold start
        _ , _ = create_rag_chain(retriever)

        # start background cleanup loop
        asyncio.create_task(_session_cleanup_loop())

        print("✅ RAG system initialized successfully")

    except Exception as e:
        print(f"❌ Error during startup: {e}")
        raise

# Ingest endpoint — re-create vector store and retriever
@app.post("/ingest", response_model=IngestResponse)
async def ingest_document():
    """Re-ingest the PDF document into the vector store"""
    global vector_store, retriever, chat_sessions

    try:
        pdf_path = os.getenv("PDF_PATH", "docs/User Query_2.pdf")
        vector_store = get_vector_store(pdf_path)
        retriever = get_retriever(vector_store)

        # Clear existing per-session chains - they reference old retriever/context
        chat_sessions.clear()

        return IngestResponse(message="PDF successfully ingested into vector store", status="success")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting document: {str(e)}")

# Helper: create per-session rag chain + memory
def _create_session(session_id: str):
    """
    Create a new chain (with its own ConversationSummaryMemory) for a session.
    This calls your existing create_rag_chain(retriever) which returns (chain, memory).
    """
    global retriever, chat_sessions
    if retriever is None:
        raise RuntimeError("Retriever not initialized")

    rag_chain, memory = create_rag_chain(retriever)
    chat_sessions[session_id] = {
        "chain": rag_chain,
        "memory": memory,
        "last_activity": time.time()
    }
    return chat_sessions[session_id]["chain"]

# Run chain.invoke in a threadpool to avoid blocking the event loop
async def _run_chain_in_thread(chain, question: str):
    loop = asyncio.get_running_loop()
    # chain.invoke is synchronous in your implementation; run it in executor
    result = await loop.run_in_executor(None, chain.invoke, question)
    return result

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint with session-based memory"""
    try:
        session_id = request.session_id or str(uuid.uuid4())

        # create session if missing
        if session_id not in chat_sessions:
            _create_session(session_id)

        session = chat_sessions[session_id]
        chain = session["chain"]

        # Run chain in background thread
        response = await _run_chain_in_thread(chain, request.question)

        # update last activity
        session["last_activity"] = time.time()

        return ChatResponse(response=response, session_id=session_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "TechnoSurge RAG API", "sessions": len(chat_sessions)}

# Background cleanup loop to free old session memory
async def _session_cleanup_loop():
    global chat_sessions
    while True:
        try:
            now = time.time()
            to_delete = []
            for sid, info in list(chat_sessions.items()):
                if now - info.get("last_activity", 0) > SESSION_TTL_SECONDS:
                    to_delete.append(sid)
            for sid in to_delete:
                # If memory object needs explicit cleanup, do it here
                del chat_sessions[sid]
                print(f"🧹 cleaned up session {sid}")
        except Exception as e:
            print(f"Session cleanup error: {e}")
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)

# Render deployment setup
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting TechnoSurge RAG API on port {port}")
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=port, 
        reload=False
    )
