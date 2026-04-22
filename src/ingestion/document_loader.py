from pathlib import Path
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def load_documents_from_dir(directory: str):
    """Carga todos los archivos .txt de un directorio."""
    docs = []
    for file_path in Path(directory).glob("*.txt"):
        logger.info(f"Cargando {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Simular estructura de documento con page_content y metadata
        doc = type(
            "Document",
            (),
            {"page_content": content, "metadata": {"source": str(file_path)}},
        )()
        docs.append(doc)
    logger.info(f"Se cargaron {len(docs)} documentos")
    return docs


def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    """Divide documentos en fragmentos (chunks) de tamaño fijo sin LangChain."""
    chunks = []
    for doc in docs:
        text = doc.page_content
        # División simple por párrafos y luego por tamaño
        paragraphs = text.split("\n\n")
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += para
            else:
                if current_chunk:
                    chunks.append(
                        type(
                            "Document",
                            (),
                            {"page_content": current_chunk, "metadata": doc.metadata},
                        )()
                    )
                current_chunk = para
        if current_chunk:
            chunks.append(
                type(
                    "Document",
                    (),
                    {"page_content": current_chunk, "metadata": doc.metadata},
                )()
            )
    logger.info(f"Se generaron {len(chunks)} chunks")
    return chunks
