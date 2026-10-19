import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Suppress deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.rag.graph import rag_graph
from app.rag.ingest import get_embedding_function

# Ensure utf-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag_evaluator")

settings = get_settings()

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = EVAL_DIR / "test_dataset.json"


def load_test_dataset(dataset_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load the benchmark question-ground-truth dataset."""
    path = dataset_path or DEFAULT_DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_ragas_evaluation(
    dataset_path: Optional[Path] = None,
    output_markdown: bool = True,
    output_json: bool = True,
) -> Dict[str, Any]:
    """
    Executes the full evaluation pipeline:
    1. Runs all test questions through the LangGraph Agentic RAG system
    2. Collects queries, generated answers, retrieved context chunks, and ground truths
    3. Evaluates 4 key Ragas metrics:
       - Faithfulness (is the answer grounded in context without hallucination?)
       - Answer Relevancy (does the answer address the question directly?)
       - Context Precision (how relevant are the top-ranked retrieved chunks?)
       - Context Recall (did the retriever find all the context needed for ground truth?)
    4. Saves detailed results to JSON and Markdown reports
    """
    test_data = load_test_dataset(dataset_path)
    logger.info("Loaded %d test question(s) for evaluation.", len(test_data))

    questions = []
    answers = []
    contexts = []
    ground_truths = []
    per_item_details = []

    print("\n" + "=" * 65)
    print("[START] AGENTIC RAG EVALUATION BENCHMARK")
    print("=" * 65)

    start_time = time.time()

    for idx, item in enumerate(test_data, start=1):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"\n[{idx}/{len(test_data)}] Evaluating: {q}")

        try:
            # Query the LangGraph Agent
            config = {"configurable": {"thread_id": f"eval-run-{idx}"}}
            result = rag_graph.invoke(
                {
                    "question": q,
                    "original_question": q,
                    "documents": [],
                    "generation": "",
                    "retry_count": 0,
                    "chat_history": [],
                    "summary": "",
                    "citations": [],
                },
                config=config,
            )

            generated_answer = result.get("generation", "No response generated.")
            retrieved_docs = result.get("documents", [])

            # Extract retrieved chunk text
            doc_texts = [d.page_content for d in retrieved_docs] if retrieved_docs else ["No context retrieved"]
            citations = result.get("citations", [])

        except Exception as exc:
            logger.error("Error executing RAG graph for query '%s': %s", q, exc)
            generated_answer = f"Error: {exc}"
            doc_texts = ["Error retrieving context"]
            citations = []

        questions.append(q)
        answers.append(generated_answer)
        contexts.append(doc_texts)
        ground_truths.append(gt)

        per_item_details.append({
            "id": item.get("id", f"q{idx}"),
            "question": q,
            "ground_truth": gt,
            "generated_answer": generated_answer,
            "retrieved_chunks_count": len(retrieved_docs),
            "citations": citations,
        })

        print(f"  -> Answer: {generated_answer[:100]}...")
        print(f"  -> Retrieved {len(retrieved_docs)} chunk(s)")

    duration = time.time() - start_time
    print(f"\nExecution finished in {duration:.1f}s. Calculating Ragas metrics...")

    # Build HuggingFace Dataset for Ragas
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    eval_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    eval_dataset = Dataset.from_dict(eval_dict)

    # Configure Groq LLM and HuggingFace Embeddings for Ragas evaluators
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    eval_llm = ChatGroq(
        model=settings.GROQ_MODEL or "llama-3.3-70b-versatile",
        api_key=settings.GROQ_API_KEY,
        temperature=0.0,
    )
    eval_embeddings = get_embedding_function()

    ragas_llm = LangchainLLMWrapper(eval_llm)
    ragas_emb = LangchainEmbeddingsWrapper(eval_embeddings)

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    for m in metrics:
        if hasattr(m, "llm"):
            m.llm = ragas_llm
        if hasattr(m, "embeddings"):
            m.embeddings = ragas_emb

    score_result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb,
    )

    # Extract score values
    scores = {}
    if hasattr(score_result, "_repr_dict"):
        raw_items = score_result._repr_dict.items()
    elif hasattr(score_result, "items"):
        raw_items = score_result.items()
    elif isinstance(score_result, dict):
        raw_items = score_result.items()
    else:
        raw_items = []

    for key, val in raw_items:
        try:
            scores[key] = round(float(val), 4)
        except Exception:
            scores[key] = val

    print("\n" + "=" * 65)
    print("[METRICS] RAGAS EVALUATION SCORES SUMMARY")
    print("=" * 65)
    for metric_name, score_val in scores.items():
        print(f"  * {metric_name.replace('_', ' ').title():<22}: {score_val}")
    print("=" * 65)

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save JSON report
    report_data = {
        "timestamp": timestamp_str,
        "model": settings.GROQ_MODEL or "llama-3.3-70b-versatile",
        "embeddings": settings.EMBEDDING_MODEL,
        "metrics": scores,
        "test_cases": per_item_details,
    }

    if output_json:
        json_path = EVAL_DIR / "eval_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"\n[Saved JSON Report]: {json_path}")

    # Generate Markdown report
    if output_markdown:
        md_path = EVAL_DIR / "EVALUATION_REPORT.md"
        md_content = generate_markdown_report(report_data, scores, duration)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"[Saved Markdown Report]: {md_path}")

    return report_data


def generate_markdown_report(data: Dict[str, Any], scores: Dict[str, Any], duration: float) -> str:
    """Generate a clean, professional GitHub-flavored Markdown report."""
    md = []
    md.append("# 📊 Enterprise Agentic RAG — Quality Evaluation Report\n")
    md.append(f"**Generated At:** `{data['timestamp']}`  \n")
    md.append(f"**LLM Evaluator:** `{data['model']}` (via Groq)  \n")
    md.append(f"**Embedding Model:** `{data['embeddings']}`  \n")
    md.append(f"**Total Benchmark Questions:** `{len(data['test_cases'])}`  \n")
    md.append(f"**Total Run Time:** `{duration:.1f}s`  \n\n")

    md.append("## 1. Metric Overview\n")
    md.append("| Metric | Score | Description |")
    md.append("| :--- | :---: | :--- |")
    md.append(f"| **Faithfulness** | **`{scores.get('faithfulness', 'N/A')}`** | Measures how factually grounded the answer is in the retrieved context (no hallucination). |")
    md.append(f"| **Answer Relevancy** | **`{scores.get('answer_relevancy', 'N/A')}`** | Measures how directly and completely the answer addresses the user question. |")
    md.append(f"| **Context Precision** | **`{scores.get('context_precision', 'N/A')}`** | Measures whether relevant chunks are ranked high at the top of the context list. |")
    md.append(f"| **Context Recall** | **`{scores.get('context_recall', 'N/A')}`** | Measures whether the retriever found all necessary context to match the ground truth. |")
    md.append("\n---\n")

    md.append("## 2. Detailed Test Cases\n")
    for idx, case in enumerate(data["test_cases"], start=1):
        md.append(f"### Q{idx}: {case['question']}\n")
        md.append(f"- **Expected (Ground Truth):** {case['ground_truth']}")
        clean_answer = case['generated_answer'].replace("\n", " ").strip()
        md.append(f"- **Agent Answer:** {clean_answer}")
        md.append(f"- **Retrieved Chunks:** `{case['retrieved_chunks_count']}` chunk(s)")
        if case.get("citations"):
            c_list = [f"`{c.get('source')}` (p. {c.get('page')})" for c in case["citations"]]
            md.append(f"- **Citations:** {', '.join(c_list)}")
        md.append("\n")

    md.append("---\n")
    md.append("## 3. How to Re-Run Evaluation\n")
    md.append("To execute this evaluation benchmark after modifying the retriever, chunking, or agent workflow:\n")
    md.append("```powershell\ncd \"enterprise-agentic-rag\"\npython -m evaluation.evaluate_rag\n```\n")

    return "\n".join(md)


if __name__ == "__main__":
    run_ragas_evaluation()
