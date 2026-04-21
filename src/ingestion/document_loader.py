import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

def load_documents_from_dir(directory: str):
    docs = []
    for file_path in Path(directory).glob("*.txt"):
        logger.info(f"Cargando {file_path}")
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs.extend(loader.load())
    logger.info(f"Se cargaron {len(docs)} documentos")
    return docs

def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    logger.info(f"Se generaron {len(chunks)} chunks")
    return chunks