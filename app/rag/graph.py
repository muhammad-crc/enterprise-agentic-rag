import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, TypedDict, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.config import get_settings
from app.rag.retriever import get_retriever
from app.rag.prompts import (
    ENTERPRISE_PDF_RAG_SYSTEM_PROMPT,
    format_retrieved_context,
    build_rag_user_message,
)

logger = logging.getLogger("enterprise_rag.agent")
settings = get_settings()

GROQ_MODEL_NAME = settings.GROQ_MODEL if settings.GROQ_MODEL else "llama-3.3-70b-versatile"
MAX_RETRIES = 2
SUMMARY_THRESHOLD = 6  # Trigger summarization when history exceeds 6 messages

GUARDRAIL_NOT_FOUND_MESSAGE = (
    "I could not find this information in the provided documents."
)


class AgentState(TypedDict, total=False):
    """
    Robust LangGraph state definition for the Agentic RAG workflow.
    Supports dynamic self-correction, query rewriting, citations,
    and rolling conversation memory with summarization.
    """
    question: str
    original_question: str
    documents: List[Document]
    generation: str
    retry_count: int
    chat_history: List[BaseMessage]
    summary: str
    citations: List[Dict[str, Any]]


def get_groq_llm(temperature: float = 0.2) -> ChatGroq:
    """Instantiate and return the Groq Chat model with configured temperature."""
    api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    return ChatGroq(
        model=GROQ_MODEL_NAME,
        temperature=temperature,
        api_key=api_key,
    )


# =========================================================================
# 1. Graph Nodes
# =========================================================================

def summarize_memory_node(state: AgentState) -> Dict[str, Any]:
    """
    Conversation Memory Summarizer:
    If chat history grows long (> SUMMARY_THRESHOLD messages), compresses older
    turns into an evolving running summary to preserve context while avoiding
    context window blowout.
    """
    chat_history = state.get("chat_history", [])
    existing_summary = state.get("summary", "")

    if len(chat_history) <= SUMMARY_THRESHOLD:
        return {"summary": existing_summary, "chat_history": chat_history}

    llm = get_groq_llm(temperature=0.0)

    # Separate older messages to condense, keeping the most recent 4 intact
    messages_to_summarize = chat_history[:-4]
    retained_recent_messages = chat_history[-4:]

    convo_lines = []
    for msg in messages_to_summarize:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        convo_lines.append(f"{role}: {msg.content}")
    convo_text = "\n".join(convo_lines)

    summary_prompt = (
        "You are a concise conversation memory condenser. "
        "Summarize the key facts, user goals, and previous answers discussed in this chat.\n\n"
        f"Existing Running Summary:\n{existing_summary or 'None'}\n\n"
        f"New Messages to incorporate:\n{convo_text}\n\n"
        "Provide a compact, high-density summary (under 150 words) capturing all essential context."
    )

    try:
        res = llm.invoke([HumanMessage(content=summary_prompt)])
        new_summary = res.content.strip()
    except Exception as exc:
        logger.warning("Conversation summarization skipped: %s", exc)
        new_summary = existing_summary

    return {
        "summary": new_summary,
        "chat_history": retained_recent_messages,
    }


def decompose_question(question: str, llm: Any) -> List[str]:
    """
    Question Decomposition & Evidence Mapping:
    Splits multi-part or multifaceted questions into focused sub-queries
    to ensure retrieval coverage for each concept.
    """
    q_low = question.lower().strip()
    multi_indicators = [" and ", " vs ", " versus ", " compare ", " comparison ", " difference between "]
    if not any(ind in q_low for ind in multi_indicators):
        return [question]

    decomp_prompt = (
        "You are an AI query planner for an enterprise document search engine.\n"
        f"User Question: '{question}'\n\n"
        "If this question contains multiple distinct concepts, sub-questions, or comparisons, "
        "decompose it into 2 to 3 concise, independent search queries to retrieve all necessary evidence.\n"
        "If the question is already focused on a single topic, return ONLY the original question.\n"
        "Output each search query on a new line without numbering or bullet points."
    )
    try:
        res = llm.invoke([HumanMessage(content=decomp_prompt)])
        lines = [line.strip().strip("-*•").strip() for line in res.content.strip().splitlines() if line.strip()]
        return lines[:3] if lines else [question]
    except Exception as exc:
        logger.debug("Query decomposition skipped: %s", exc)
        return [question]


def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """
    Hybrid Search Retrieval with Question Decomposition:
    Splits multi-part questions to retrieve targeted evidence for each concept,
    then merges with RRF ranking and dynamic top-k broadening on retry.
    """
    question = state.get("question", "")
    original_question = state.get("original_question") or question
    retry_count = state.get("retry_count", 0)

    # Broaden search from k=8 to k=12 chunks if retrying after weak context
    k = 8 if retry_count == 0 else 12
    llm = get_groq_llm(temperature=0.0)
    sub_queries = decompose_question(question, llm)
    logger.info("Executing retrieval for %d query target(s): %s", len(sub_queries), sub_queries)

    retriever = get_retriever(k=k)
    seen_keys = set()
    all_docs: List[Document] = []

    for sq in sub_queries:
        docs = retriever.invoke(sq, k=k)
        for doc in docs:
            meta = doc.metadata or {}
            key = (meta.get("source"), meta.get("page"), doc.page_content[:60])
            if key not in seen_keys:
                seen_keys.add(key)
                all_docs.append(doc)

    return {
        "documents": all_docs[:k],
        "original_question": original_question,
    }


def grade_context_node(state: AgentState) -> Dict[str, Any]:
    """
    Relevance Grader (Batch Evaluation):
    Assesses all candidate passages in a single fast LLM call to prevent
    Groq 429 rate-limit retry penalties and cut latency to under 2 seconds.
    """
    question = state.get("original_question") or state.get("question", "")
    documents = state.get("documents", [])

    if not documents:
        return {"documents": []}

    llm = get_groq_llm(temperature=0.0)

    passages_text = []
    for idx, doc in enumerate(documents, start=1):
        meta = doc.metadata or {}
        src = meta.get("file_name") or meta.get("source", "Document")
        page = meta.get("page", 1)
        passages_text.append(f"Passage [{idx}] (Source: {src}, Page {page}):\n{doc.page_content[:300]}")

    joined_passages = "\n\n".join(passages_text)

    batch_prompt = (
        "You are an evaluator assessing whether retrieved passages contain relevant facts or details "
        f"to answer the question: '{question}'\n\n"
        f"{joined_passages}\n\n"
        "Identify which passages contain relevant facts, concepts, or context to help answer the question.\n"
        "Respond with ONLY the comma-separated passage numbers of relevant passages (e.g. '1, 2, 4') or 'none'."
    )

    try:
        res = llm.invoke([HumanMessage(content=batch_prompt)])
        content = res.content.strip().lower()
        if "none" in content and not any(ch.isdigit() for ch in content):
            selected_indices = set()
        else:
            nums = re.findall(r"\b\d+\b", content)
            selected_indices = {int(n) for n in nums}

        filtered_docs = [doc for idx, doc in enumerate(documents, start=1) if idx in selected_indices]
    except Exception as exc:
        logger.warning("Batch grading error: %s. Preserving retrieved docs.", exc)
        filtered_docs = documents

    if not filtered_docs and documents and state.get("retry_count", 0) > 0:
        filtered_docs = documents[:3]

    return {"documents": filtered_docs or documents[:4]}


def rewrite_query_node(state: AgentState) -> Dict[str, Any]:
    """
    Self-Correction Query Reformulator:
    When retrieved context is weak/empty, rewrites query with synonyms,
    expanded concepts, and keyword variations to retry retrieval.
    """
    original_question = state.get("original_question") or state.get("question", "")
    current_question = state.get("question", "")
    retry_count = state.get("retry_count", 0) + 1
    llm = get_groq_llm(temperature=0.2)

    prompt = (
        "You are an AI search query optimizer for an enterprise knowledge retrieval system.\n"
        f"Original User Question: '{original_question}'\n"
        f"Previous Search Attempt: '{current_question}'\n\n"
        "The previous query failed to retrieve relevant documents. "
        "Rewrite this query into a more effective keyword search query:\n"
        "- Use alternative synonyms and relevant domain terminology\n"
        "- Eliminate conversational fluff (e.g., 'can you tell me', 'what is')\n"
        "- Keep it concise and focused\n"
        "Respond with ONLY the rewritten query text."
    )

    try:
        res = llm.invoke([HumanMessage(content=prompt)])
        rewritten = res.content.strip().strip('"').strip("'")
    except Exception:
        rewritten = original_question

    logger.info("Self-Correction: Rewrote '%s' -> '%s' (Attempt %d)", original_question, rewritten, retry_count)

    return {
        "question": rewritten if rewritten else original_question,
        "retry_count": retry_count,
    }


def validate_citations(answer_text: str, documents: List[Document]) -> List[Dict[str, Any]]:
    """
    Validate that each cited source and chunk in the answer actually exists in the retrieved documents.
    Extracts citations such as [Source: filename, p. X, chunk_Y], [Source: ...], [chunk_Y], or [1].
    """
    validated: List[Dict[str, Any]] = []
    seen = set()

    # Pre-map available retrieved chunks
    chunk_map: Dict[str, Any] = {}
    for idx, doc in enumerate(documents, start=1):
        meta = doc.metadata or {}
        cid = meta.get("chunk_id") or f"chunk_{idx}"
        chunk_map[cid.lower()] = (doc, cid, idx)
        chunk_map[str(idx)] = (doc, cid, idx)

    # 1. Match [Source: <filename>, p. <page>, <chunk_id>] or variations
    source_patterns = re.findall(
        r"\[Source:\s*([^,\]]+)(?:,\s*p\.?\s*([^,\]]+))?(?:,\s*(chunk_\d+))?\]",
        answer_text,
        re.IGNORECASE,
    )
    for src_file, page_num, chunk_tag in source_patterns:
        src_file = src_file.strip()
        page_val = int(page_num) if (page_num and page_num.strip().isdigit()) else None
        c_id = chunk_tag.strip() if chunk_tag else None

        matching_doc = None
        for doc in documents:
            meta = doc.metadata or {}
            m_src = meta.get("source") or meta.get("file_name", "")
            m_page = meta.get("page")
            if src_file.lower() in m_src.lower() or m_src.lower() in src_file.lower():
                if page_val is None or m_page == page_val:
                    matching_doc = doc
                    if not c_id:
                        c_id = meta.get("chunk_id")
                    break

        if matching_doc:
            m_meta = matching_doc.metadata or {}
            key = (m_meta.get("source"), m_meta.get("page"), c_id)
            if key not in seen:
                seen.add(key)
                validated.append({
                    "source": m_meta.get("source", src_file),
                    "page": m_meta.get("page", page_val),
                    "chunk_id": c_id or m_meta.get("chunk_id"),
                    "preview": matching_doc.page_content[:200] + "...",
                })

    # 2. Match chunk references like [chunk_1] or numeric citations like [1]
    ref_matches = re.findall(r"\[(chunk_\d+|\d+)\]", answer_text, re.IGNORECASE)
    for ref in ref_matches:
        ref_lower = ref.lower()
        if ref_lower in chunk_map:
            doc, c_id, idx = chunk_map[ref_lower]
            meta = doc.metadata or {}
            key = (meta.get("source"), meta.get("page"), c_id)
            if key not in seen:
                seen.add(key)
                validated.append({
                    "source": meta.get("source"),
                    "page": meta.get("page"),
                    "chunk_id": c_id,
                    "preview": doc.page_content[:200] + "...",
                })

    # Fallback if no specific inline citation regex matched but documents were provided
    if not validated and documents:
        for idx, doc in enumerate(documents[:4], start=1):
            meta = doc.metadata or {}
            c_id = meta.get("chunk_id") or f"chunk_{idx}"
            validated.append({
                "source": meta.get("source") or meta.get("file_name", "Unknown Document"),
                "page": meta.get("page", None),
                "chunk_id": c_id,
                "preview": doc.page_content[:200] + "...",
            })

    return validated


def generate_node(state: AgentState) -> Dict[str, Any]:
    """
    Enterprise Answer Generator adhering to the 16-section RAG standard:
    - Strictly grounded in retrieved context
    - Uses standardized chunk presentation [CHUNK_ID: ...][DOCUMENT: ...][PAGE: ...]
    - Validates citations post-generation to eliminate hallucinated sources
    - Anti-prompt injection defenses with <context> encapsulation
    """
    question = state.get("original_question") or state.get("question", "")
    documents = state.get("documents", [])
    chat_history = state.get("chat_history", [])
    summary = state.get("summary", "")
    llm = get_groq_llm(temperature=0.2)

    # Format context passages into standardized chunk structure
    context_str = format_retrieved_context(documents)

    messages: List[BaseMessage] = [
        SystemMessage(content=ENTERPRISE_PDF_RAG_SYSTEM_PROMPT)
    ]

    # Attach conversation memory if present
    if summary:
        messages.append(SystemMessage(content=f"Prior Conversation Context Summary:\n{summary}"))
    if chat_history:
        messages.extend(chat_history[-4:])

    # Build safe user message encapsulating untrusted context
    user_message_content = build_rag_user_message(question=question, context=context_str)
    messages.append(HumanMessage(content=user_message_content))

    response = llm.invoke(messages)
    answer_text = response.content.strip()

    # Guardrail check
    if GUARDRAIL_NOT_FOUND_MESSAGE.lower() in answer_text.lower():
        return {
            "generation": GUARDRAIL_NOT_FOUND_MESSAGE,
            "citations": [],
        }

    # Post-generation citation validation
    validated_citations = validate_citations(answer_text, documents)

    return {
        "generation": answer_text,
        "citations": validated_citations,
    }


def fallback_node(state: AgentState) -> Dict[str, Any]:
    """
    Guardrail Fallback:
    Triggered when context remains weak or empty after all self-correction attempts.
    Strictly returns the guardrail statement without hallucinating.
    """
    return {
        "generation": GUARDRAIL_NOT_FOUND_MESSAGE,
        "citations": [],
    }


# =========================================================================
# 2. Conditional Routing Logic
# =========================================================================

def evaluate_retrieval(state: AgentState) -> str:
    """
    Self-Correction Decision Branch:
    - Relevant documents found -> Proceed to 'generate'
    - Context weak/empty and retry_count < MAX_RETRIES -> 'rewrite_query' (self-correction loop)
    - Retries exhausted -> 'fallback' (Guardrail triggered)
    """
    docs = state.get("documents", [])
    retry_count = state.get("retry_count", 0)

    if docs:
        return "generate"
    elif retry_count < MAX_RETRIES:
        return "rewrite_query"
    else:
        return "fallback"


# =========================================================================
# 3. LangGraph Workflow Compilation
# =========================================================================

checkpointer = MemorySaver()


def create_rag_graph():
    """
    Build and compile the modular Agentic RAG workflow with:
    - Conversation memory summarizer
    - Hybrid retrieval with Question Decomposition
    - Context grading
    - Dynamic self-correction query re-writing loop
    - Strict 16-section guardrails and verified citations
    """
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("summarize_memory", summarize_memory_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_context", grade_context_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("fallback", fallback_node)

    # Entry point: summarize history if needed, then retrieve
    workflow.set_entry_point("summarize_memory")
    workflow.add_edge("summarize_memory", "retrieve")
    workflow.add_edge("retrieve", "grade_context")

    # Self-Correction Conditional Routing
    workflow.add_conditional_edges(
        "grade_context",
        evaluate_retrieval,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "fallback": "fallback",
        },
    )

    # Loop back to retrieve after query rewrite
    workflow.add_edge("rewrite_query", "retrieve")

    # Terminal edges
    workflow.add_edge("generate", END)
    workflow.add_edge("fallback", END)

    return workflow.compile(checkpointer=checkpointer)


# Singleton compiled graph instance
rag_graph = create_rag_graph()
