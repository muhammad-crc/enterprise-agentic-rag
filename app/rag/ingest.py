import logging
import os
import sys
import warnings
from pathlib import Path
from functools import lru_cache
from typing import Any, List, Optional, Union, Dict

# Suppress LangChain deprecation notices globally for clean console output
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader

# Lazy/conditional imports for vector stores
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Support running directly or as a module
try:
    from app.config import get_settings
except ImportError:
    # Handle direct script execution: python app/rag/ingest.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from app.config import get_settings

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("enterprise_rag.ingest")

settings = get_settings()


@lru_cache(maxsize=1)
def get_embedding_function():
    """
    Initialize and return configured embeddings (HuggingFace/local or OpenAI).
    """
    if settings.EMBEDDING_PROVIDER == "openai" or (
        settings.OPENAI_API_KEY and settings.EMBEDDING_PROVIDER != "huggingface"
    ):
        api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY is not set.")
        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL if "embedding" in settings.EMBEDDING_MODEL else "text-embedding-3-small",
            openai_api_key=api_key,
        )

    # FastEmbed / Local Embeddings (Ultra-lightweight ONNX runtime ~120MB RAM, zero PyTorch overhead)
    # Uses the exact same model weights (e.g. sentence-transformers/all-MiniLM-L6-v2) with identical 384-dim vectors
    try:
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
        logger.info("Loading FastEmbed embeddings (model: %s)...", settings.EMBEDDING_MODEL)
        return FastEmbedEmbeddings(model_name=settings.EMBEDDING_MODEL)
    except Exception as fe_err:
        logger.warning("FastEmbed not loaded (%s), attempting HuggingFace fallback.", fe_err)

    # HuggingFace fallback (Standard PyTorch transformer)
    try:
        # Compatibility stub for Windows environments where pyarrow DLL is blocked by Application Control policies
        if "datasets" not in sys.modules:
            try:
                import datasets  # noqa: F401
            except Exception:
                import types
                import importlib.machinery
                class _DatasetsStub(types.ModuleType):
                    def __getattr__(self, name):
                        if name.startswith("__") and name.endswith("__"):
                            raise AttributeError(name)
                        cls = type(name, (), {})
                        setattr(self, name, cls)
                        return cls
                stub = _DatasetsStub("datasets")
                stub.__spec__ = importlib.machinery.ModuleSpec("datasets", None)
                stub.__version__ = "3.0.0"
                sys.modules["datasets"] = stub

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    except Exception as e:
        logger.error("Failed to load embeddings: %s", e)
        raise RuntimeError(
            f"Could not load embedding model '{settings.EMBEDDING_MODEL}'. Error: {e}"
        )


def load_single_pdf(file_path: Union[str, Path]) -> List[Document]:
    """
    Load a single PDF using PyMuPDF with fallback to pypdf.
    Ensures standard metadata fields (source, file_name, page).
    """
    path_obj = Path(file_path)
    if not path_obj.is_file():
        logger.error("File does not exist: %s", path_obj)
        return []

    documents: List[Document] = []

    # Attempt PyMuPDF (faster and preserves layout structure well)
    try:
        loader = PyMuPDFLoader(str(path_obj))
        documents = loader.load()
        logger.info("Loaded '%s' (%d pages) using PyMuPDF.", path_obj.name, len(documents))
    except Exception as exc_pymupdf:
        logger.warning(
            "PyMuPDF failed on '%s' (%s). Falling back to PyPDFLoader...",
            path_obj.name,
            exc_pymupdf,
        )
        try:
            loader = PyPDFLoader(str(path_obj))
            documents = loader.load()
            logger.info("Loaded '%s' (%d pages) using PyPDF.", path_obj.name, len(documents))
        except Exception as exc_pypdf:
            logger.error("Failed to load PDF '%s' with both loaders: %s", path_obj.name, exc_pypdf)
            return []

    # Guarantee standardized metadata fields (clean file_name and 1-indexed page number)
    for doc in documents:
        raw_page = doc.metadata.get("page", 0)
        page_number = int(raw_page) + 1 if isinstance(raw_page, int) else raw_page

        doc.metadata.update(
            {
                "source": path_obj.name,  # Clean file name for display and filtering
                "file_path": str(path_obj.resolve()),
                "file_name": path_obj.name,
                "page": page_number,
            }
        )

    return documents


def load_pdfs_from_directory(folder_path: Union[str, Path]) -> List[Document]:
    """
    Discover and load all PDF files within the given directory.
    """
    directory = Path(folder_path)
    if not directory.exists():
        logger.warning("Directory '%s' does not exist. Creating it...", directory)
        directory.mkdir(parents=True, exist_ok=True)
        return []

    pdf_files = list(directory.glob("**/*.pdf"))
    if not pdf_files:
        logger.info("No PDF files found in '%s'.", directory)
        return []

    logger.info("Discovered %d PDF file(s) in '%s'. Starting load...", len(pdf_files), directory)
    all_documents: List[Document] = []

    for pdf_path in pdf_files:
        try:
            docs = load_single_pdf(pdf_path)
            all_documents.extend(docs)
        except Exception as exc:
            logger.error("Unexpected error processing file '%s': %s", pdf_path, exc, exc_info=True)

    logger.info("Total pages loaded from %d file(s): %d", len(pdf_files), len(all_documents))
    return all_documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Improved semantic chunking:
    Uses hierarchical separators (paragraphs -> sentence boundaries -> words)
    to prevent cutting thoughts in half and preserve semantic coherence.
    """
    if not documents:
        logger.warning("No documents provided to split.")
        return []

    # Semantic boundary separators in order of precedence
    semantic_separators = [
        "\n\n\n",
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        "; ",
        ", ",
        " ",
        "",
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=semantic_separators,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)

    # Filter out empty or trivially short header noise (e.g. repeated titles under 50 chars)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= 50]

    # Attach chunk metadata (chunk_id, chunk_index, character length, source, page)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{idx + 1}"
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["char_count"] = len(chunk.page_content)
        # Ensure source and page are explicitly present
        if "source" not in chunk.metadata:
            chunk.metadata["source"] = chunk.metadata.get("file_name", "unknown")
        if "page" not in chunk.metadata:
            chunk.metadata["page"] = 1

    logger.info(
        "Split %d document page(s) into %d semantic chunks (chunk_size=%d, chunk_overlap=%d).",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


def ingest_to_vector_store(
    documents: List[Document],
    vector_store_type: Optional[str] = None,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> Optional[Any]:
    """
    Splits documents and indexes them into the selected vector store (Chroma or Qdrant).
    Seamless switching via `vector_store_type` argument or `VECTOR_STORE_TYPE` in config.
    """
    if not documents:
        logger.warning("No documents available for ingestion.")
        return None

    target_store = (vector_store_type or settings.VECTOR_STORE_TYPE).lower()
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embeddings = get_embedding_function()

    try:
        # --- QDRANT VECTOR STORE ---
        if target_store == "qdrant":
            try:
                from qdrant_client import QdrantClient
                from langchain_qdrant import QdrantVectorStore as Qdrant
            except ImportError:
                raise ImportError(
                    "Qdrant integration packages missing. Install: pip install langchain-qdrant qdrant-client"
                )

            logger.info("Connecting to Qdrant at %s...", settings.QDRANT_URL)
            client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)

            vector_store = Qdrant.from_documents(
                documents=chunks,
                embedding=embeddings,
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY or None,
                collection_name=settings.COLLECTION_NAME,
            )
            logger.info("Successfully indexed %d chunks into Qdrant collection '%s'.", len(chunks), settings.COLLECTION_NAME)
            return vector_store

        # --- CHROMA VECTOR STORE (Default) ---
        logger.info(
            "Indexing %d chunks into ChromaDB at '%s' (collection: '%s')...",
            len(chunks),
            settings.CHROMA_PERSIST_DIR,
            settings.COLLECTION_NAME,
        )
        os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR,
            collection_name=settings.COLLECTION_NAME,
        )
        logger.info("Successfully persisted %d chunks into ChromaDB.", len(chunks))
        return vector_store

    except Exception as exc:
        logger.critical(
            "Failed vector store ingestion (%s): %s",
            target_store,
            exc,
            exc_info=True,
        )
        raise exc


# Alias for backward compatibility
load_documents_from_directory = load_pdfs_from_directory
get_embeddings = get_embedding_function


def run_pipeline(
    data_dir: Optional[Union[str, Path]] = None,
    vector_store_type: Optional[str] = None,
):
    """
    High-level orchestrator for reading PDFs, chunking, and indexing.
    """
    base_data_dir = (
        Path(data_dir)
        if data_dir
        else Path(__file__).resolve().parent.parent.parent / "data"
    )

    logger.info("Starting ingestion pipeline for folder: %s", base_data_dir)
    docs = load_pdfs_from_directory(base_data_dir)

    if not docs:
        logger.info("Pipeline completed: No documents to process in '%s'.", base_data_dir)
        return None

    vector_store = ingest_to_vector_store(
        documents=docs,
        vector_store_type=vector_store_type,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    logger.info("Ingestion pipeline completed successfully.")
    return vector_store


if __name__ == "__main__":
    run_pipeline()
