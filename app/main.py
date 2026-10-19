import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage

from app.config import get_settings
from app.rag.graph import rag_graph
from app.rag.ingest import load_documents_from_directory, ingest_to_vector_store

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("enterprise_rag.api")

settings = get_settings()

# Initialize FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="Enterprise Agentic RAG API powered by LangGraph, FastAPI, ChromaDB / Qdrant, and Groq",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows Next.js frontend or any client origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Document storage directory
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# In-memory session store: session_id -> list of LangChain BaseMessages
SESSION_HISTORIES: Dict[str, List[Any]] = {}


# =========================================================================
# Pydantic Request & Response Models
# =========================================================================

class HistoryMessage(BaseModel):
    role: str = Field(..., description="Message author role: 'user' or 'assistant'")
    content: str = Field(..., description="Text content of the message")


class QueryRequest(BaseModel):
    query: str = Field(..., description="User question or query", min_length=1)
    session_id: Optional[str] = Field(default="default-session", description="Unique conversation session ID")
    conversation_history: Optional[List[HistoryMessage]] = Field(
        default=None,
        description="Optional list of previous messages sent from client",
    )


class DocumentSource(BaseModel):
    source: str = Field(..., description="File name of the source document")
    page: Optional[int] = Field(default=None, description="1-indexed page number")
    preview: str = Field(..., description="Relevant text excerpt snippet")
    chunk_id: Optional[str] = Field(default=None, description="Unique chunk identifier")


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[DocumentSource] = []
    session_id: str


class IngestResponse(BaseModel):
    status: str
    files_processed: List[str]
    message: str


class HealthResponse(BaseModel):
    status: str
    app_name: str
    llm_provider: str
    vector_store: str


# =========================================================================
# API Endpoints
# =========================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint to verify system status and configuration."""
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        llm_provider=settings.LLM_PROVIDER,
        vector_store=settings.VECTOR_STORE_TYPE,
    )


# --- Chat Endpoints (Supporting both /chat and /api/v1/chat for frontend compatibility) ---

@app.post("/chat", response_model=QueryResponse, tags=["RAG"])
@app.post("/api/v1/chat", response_model=QueryResponse, tags=["RAG"])
async def chat_endpoint(request: QueryRequest):
    """
    Execute LangGraph Agentic RAG workflow:
    - Ingests user question and conversation history
    - Performs hybrid retrieval (dense embeddings + BM25 keyword matching)
    - Grades context relevance with self-correction query re-writing
    - Generates grounded answer with exact file name + page citations
    """
    try:
        session_id = request.session_id or "default-session"
        session_data = SESSION_HISTORIES.setdefault(session_id, {"messages": [], "summary": ""})
        if isinstance(session_data, list):
            # Backward compatibility migration if stored as list
            session_data = {"messages": session_data, "summary": ""}
            SESSION_HISTORIES[session_id] = session_data

        stored_history = session_data["messages"]
        current_summary = session_data["summary"]

        # Build conversation memory: merge any client-supplied history with server session
        history_messages = list(stored_history)
        if request.conversation_history:
            history_messages = []
            for item in request.conversation_history:
                if item.role.lower() == "user":
                    history_messages.append(HumanMessage(content=item.content))
                else:
                    history_messages.append(AIMessage(content=item.content))

        # Initial state for LangGraph workflow
        initial_state = {
            "question": request.query,
            "original_question": request.query,
            "documents": [],
            "generation": "",
            "retry_count": 0,
            "chat_history": history_messages,
            "summary": current_summary,
            "citations": [],
        }

        # Invoke LangGraph agent with thread checkpointing
        config = {"configurable": {"thread_id": session_id}}
        result = rag_graph.invoke(initial_state, config=config)

        answer = result.get("generation", "No response generated.")

        # Update in-memory session history and rolling summary
        updated_history = result.get("chat_history", history_messages)
        updated_history.append(HumanMessage(content=request.query))
        updated_history.append(AIMessage(content=answer))
        session_data["messages"] = updated_history
        session_data["summary"] = result.get("summary", current_summary)

        # Format sources with standardized filename, page, and snippet preview
        sources: List[DocumentSource] = []
        if result.get("citations"):
            for cite in result["citations"]:
                sources.append(
                    DocumentSource(
                        source=cite.get("source", "unknown"),
                        page=cite.get("page", None),
                        preview=cite.get("preview", ""),
                        chunk_id=cite.get("chunk_id", None),
                    )
                )
        else:
            for doc in result.get("documents", []):
                meta = doc.metadata or {}
                sources.append(
                    DocumentSource(
                        source=meta.get("source") or meta.get("file_name", "unknown"),
                        page=meta.get("page", None),
                        preview=doc.page_content[:200] + "...",
                        chunk_id=meta.get("chunk_id", None),
                    )
                )

        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            session_id=session_id,
        )

    except Exception as exc:
        logger.error("Chat execution error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent workflow error: {str(exc)}",
        )


# --- Upload & Ingestion Endpoints (Supporting both /upload and /api/v1/ingest/upload) ---

@app.post("/upload", response_model=IngestResponse, tags=["Ingestion"])
@app.post("/api/v1/ingest/upload", response_model=IngestResponse, tags=["Ingestion"])
async def upload_and_ingest(files: List[UploadFile] = File(...)):
    """
    Upload one or more PDF files and ingest into the vector database:
    - Validates PDF format
    - Saves files to the storage directory
    - Chunks text semantically
    - Indexes into ChromaDB / Qdrant with clean metadata
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided.",
        )

    processed_files = []
    try:
        for file in files:
            if not file.filename or not file.filename.lower().endswith(".pdf"):
                logger.warning("Skipping unsupported non-PDF file: %s", file.filename)
                continue

            destination = DATA_DIR / file.filename
            with open(destination, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            processed_files.append(file.filename)
            logger.info("Saved uploaded PDF: %s", file.filename)

        if not processed_files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid PDF documents found in upload.",
            )

        # Trigger chunking and vector store indexing
        documents = load_documents_from_directory(str(DATA_DIR))
        ingest_to_vector_store(documents)

        return IngestResponse(
            status="success",
            files_processed=processed_files,
            message=f"Successfully ingested {len(processed_files)} document(s) into knowledge base.",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Ingestion error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(exc)}",
        )


# =========================================================================
# Application Runner
# =========================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG,
    )
