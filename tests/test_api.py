import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Importar la app después de parchear las dependencias para evitar inicializaciones no deseadas
with patch('src.api.rag_chain.get_groq_api_key', return_value='fake-key'):
    with patch('src.api.rag_chain.get_pinecone_api_key', return_value='fake-key'):
        from src.api.main import app

client = TestClient(app)

def test_ask_endpoint_success():
    # Parchear la función ask en el módulo main (donde se usa)
    with patch('src.api.main.ask') as mock_ask:
        mock_ask.return_value = ("Respuesta generada", ["Fuente 1", "Fuente 2"])
        response = client.post("/ask", json={"text": "¿Pregunta de prueba?"})
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Respuesta generada"
        assert data["sources"] == ["Fuente 1", "Fuente 2"]
        mock_ask.assert_called_once_with("¿Pregunta de prueba?")

def test_ask_endpoint_empty_question():
    response = client.post("/ask", json={"text": ""})
    assert response.status_code == 400
    assert response.json()["detail"] == "Pregunta vacía"
