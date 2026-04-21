import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.api.main import app

client = TestClient(app)

@patch('src.api.rag_chain.ask')
def test_ask_endpoint_success(mock_ask):
    # Simular la respuesta de la función ask
    mock_ask.return_value = ("Respuesta generada", ["Fuente 1", "Fuente 2"])
    
    response = client.post("/ask", json={"text": "¿Pregunta de prueba?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Respuesta generada"
    assert data["sources"] == ["Fuente 1", "Fuente 2"]
    mock_ask.assert_called_once_with("¿Pregunta de prueba?")

@patch('src.api.rag_chain.ask')
def test_ask_endpoint_empty_question(mock_ask):
    response = client.post("/ask", json={"text": ""})
    assert response.status_code == 400
    assert response.json()["detail"] == "Pregunta vacía"
    mock_ask.assert_not_called()
