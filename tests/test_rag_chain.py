import pytest
from unittest.mock import patch, MagicMock
from src.api.rag_chain import ask

@patch('src.api.rag_chain.get_groq_api_key')
@patch('src.api.rag_chain.get_pinecone_api_key')
@patch('src.api.rag_chain.retrieve')
@patch('src.api.rag_chain.generate_answer')
def test_ask_with_context(mock_generate, mock_retrieve, mock_pinecone_key, mock_groq_key):
    mock_pinecone_key.return_value = "fake-pinecone-key"
    mock_groq_key.return_value = "fake-groq-key"
    mock_retrieve.return_value = ["Contexto A", "Contexto B"]
    mock_generate.return_value = "Respuesta generada"
    
    answer, sources = ask("¿Pregunta?")
    assert answer == "Respuesta generada"
    assert sources == ["Contexto A", "Contexto B"]
    mock_retrieve.assert_called_once_with("¿Pregunta?")
    mock_generate.assert_called_once_with("¿Pregunta?", ["Contexto A", "Contexto B"])

@patch('src.api.rag_chain.get_groq_api_key')
@patch('src.api.rag_chain.get_pinecone_api_key')
@patch('src.api.rag_chain.retrieve')
def test_ask_no_context(mock_retrieve, mock_pinecone_key, mock_groq_key):
    mock_pinecone_key.return_value = "fake-pinecone-key"
    mock_groq_key.return_value = "fake-groq-key"
    mock_retrieve.return_value = []
    
    answer, sources = ask("Sin contexto")
    assert answer == "No encontré información relevante en los documentos."
    assert sources == []
