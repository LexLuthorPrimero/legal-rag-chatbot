#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.document_loader import load_documents_from_dir, chunk_documents
from src.ingestion.embeddings_and_pinecone import embed_and_upload   # <-- DESCOMENTADO
from src.utils.logging import setup_logger

logger = setup_logger("ingestion")

def main():
    logger.info("Iniciando pipeline de ingesta de documentos")
    docs_dir = "data/raw/leyes"
    logger.info(f"Cargando documentos desde {docs_dir}")
    docs = load_documents_from_dir(docs_dir)
    if not docs:
        logger.error("No se encontraron documentos. Abortando.")
        return
    logger.info("Dividiendo en chunks...")
    chunks = chunk_documents(docs)
    logger.info("Generando embeddings y subiendo a Pinecone...")
    embed_and_upload(chunks)   # <-- DESCOMENTADO
    logger.info("Ingesta finalizada.")

if __name__ == "__main__":
    main()