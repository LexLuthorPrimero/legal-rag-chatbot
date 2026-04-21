import pytest
from unittest.mock import patch, MagicMock
from src.api.rag_chain import ask

@patch('src.api.rag_chain.retrieve')
@patch('src.api.rag_chain.generate_answer')
def test_ask_with_context(mock_generate, mock_retrieve):
    # Simular que retrieve devuelve contextos
    mock_retrieve.return_value = ["Contexto de prueba 1", "Contexto de prueba 2"]
    mock_generate.return_value = "Respuesta generada"
    
    answer, sources = ask("¿Pregunta de prueba?")
    
    assert answer == "Respuesta generada"
    assert sources == ["Contexto de prueba 1", "Contexto de prueba 2"]
    mock_retrieve.assert_called_once_with("¿Pregunta de prueba?")
    mock_generate.assert_called_once_with("¿Pregunta de prueba?", ["Contexto de prueba 1", "Contexto de prueba 2"])

@patch('src.api.rag_chain.retrieve')
def test_ask_no_context(mock_retrieve):
    mock_retrieve.return_value = []
    
    answer, sources = ask("Pregunta sin contexto")
    
    assert answer == "No encontré información relevante en los documentos."
    assert sources == []
