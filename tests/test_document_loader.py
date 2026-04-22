import tempfile
from pathlib import Path
from src.ingestion.document_loader import load_documents_from_dir, chunk_documents


def test_load_documents_from_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = Path(tmpdir) / "doc1.txt"
        file1.write_text("Contenido del documento 1", encoding="utf-8")
        file2 = Path(tmpdir) / "doc2.txt"
        file2.write_text("Contenido del documento 2", encoding="utf-8")
        docs = load_documents_from_dir(tmpdir)
        assert len(docs) == 2
        contents = [doc.page_content for doc in docs]
        assert "Contenido del documento 1" in contents
        assert "Contenido del documento 2" in contents


def test_chunk_documents():
    with tempfile.TemporaryDirectory() as tmpdir:
        file = Path(tmpdir) / "doc.txt"
        file.write_text("Este es un texto largo. " * 100, encoding="utf-8")
        docs = load_documents_from_dir(tmpdir)
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1
        assert all(len(chunk.page_content) > 0 for chunk in chunks)
