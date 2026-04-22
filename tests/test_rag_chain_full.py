import pytest
from unittest.mock import patch, MagicMock
import src.api.rag_chain  # para acceder a _groq_client
from src.api.rag_chain import get_groq_client, get_pinecone_index, retrieve, generate_answer, ask

# Helper para simular arrays con .tolist()
class ArrayMock:
    def __init__(self, data):
        self.data = data
    def tolist(self):
        return self.data

# ------------------------------------------------------------
# Tests para get_groq_client
# ------------------------------------------------------------
@patch('src.api.rag_chain.get_groq_api_key')
def test_get_groq_client_success(mock_get_key):
    src.api.rag_chain._groq_client = None
    mock_get_key.return_value = "fake-key"
    client = get_groq_client()
    assert client is not None
    client2 = get_groq_client()
    assert client is client2

@patch('src.api.rag_chain.get_groq_api_key')
def test_get_groq_client_no_key(mock_get_key):
    src.api.rag_chain._groq_client = None
    mock_get_key.return_value = None
    client = get_groq_client()
    assert client is None

# ------------------------------------------------------------
# Tests para get_pinecone_index
# ------------------------------------------------------------
@patch('src.api.rag_chain.pinecone.Pinecone')
@patch('src.api.rag_chain.get_pinecone_api_key')
@patch('src.api.rag_chain.get_pinecone_environment')
@patch('src.api.rag_chain.get_pinecone_index_name')
def test_get_pinecone_index(mock_idx_name, mock_env, mock_key, mock_pinecone):
    mock_key.return_value = "key"
    mock_env.return_value = "env"
    mock_idx_name.return_value = "idx"
    mock_index = MagicMock()
    mock_pinecone.return_value.Index.return_value = mock_index

    idx = get_pinecone_index()
    assert idx == mock_index
    mock_pinecone.assert_called_once_with(api_key="key", environment="env")
    mock_pinecone.return_value.Index.assert_called_once_with("idx")

# ------------------------------------------------------------
# Tests para retrieve
# ------------------------------------------------------------
@patch('src.api.rag_chain.embedding_model')
@patch('src.api.rag_chain.get_pinecone_index')
def test_retrieve_success(mock_get_index, mock_embed_model):
    mock_embed_model.embed.return_value = (ArrayMock([0.1] * 384),)
    mock_index = MagicMock()
    mock_index.query.return_value = {
        'matches': [
            {'metadata': {'text': 'contexto 1'}},
            {'metadata': {'text': 'contexto 2'}}
        ]
    }
    mock_get_index.return_value = mock_index

    contexts = retrieve("pregunta test", top_k=2)
    assert contexts == ["contexto 1", "contexto 2"]
    mock_embed_model.embed.assert_called_once_with(["pregunta test"])
    mock_index.query.assert_called_once()

@patch('src.api.rag_chain.embedding_model')
@patch('src.api.rag_chain.get_pinecone_index')
def test_retrieve_no_matches(mock_get_index, mock_embed_model):
    mock_embed_model.embed.return_value = (ArrayMock([0.1] * 384),)
    mock_index = MagicMock()
    mock_index.query.return_value = {'matches': []}
    mock_get_index.return_value = mock_index

    contexts = retrieve("pregunta")
    assert contexts == []

# ------------------------------------------------------------
# Tests para generate_answer
# ------------------------------------------------------------
@patch('src.api.rag_chain.get_groq_client')
def test_generate_answer_with_client(mock_get_client):
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = "respuesta generada"
    mock_client.chat.completions.create.return_value = mock_completion
    mock_get_client.return_value = mock_client

    answer = generate_answer("pregunta", ["contexto1"])
    assert answer == "respuesta generada"
    mock_client.chat.completions.create.assert_called_once()

@patch('src.api.rag_chain.get_groq_client')
def test_generate_answer_no_client(mock_get_client):
    mock_get_client.return_value = None
    answer = generate_answer("pregunta", ["contexto"])
    assert "simulada" in answer

@patch('src.api.rag_chain.get_groq_client')
def test_generate_answer_client_error(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API error")
    mock_get_client.return_value = mock_client

    answer = generate_answer("pregunta", ["contexto"])
    assert "Lo siento" in answer

# ------------------------------------------------------------
# Tests para ask
# ------------------------------------------------------------
@patch('src.api.rag_chain.retrieve')
@patch('src.api.rag_chain.generate_answer')
def test_ask_with_context(mock_generate, mock_retrieve):
    mock_retrieve.return_value = ["contexto1", "contexto2"]
    mock_generate.return_value = "respuesta"
    answer, sources = ask("pregunta")
    assert answer == "respuesta"
    assert sources == ["contexto1", "contexto2"]

@patch('src.api.rag_chain.retrieve')
def test_ask_no_context(mock_retrieve):
    mock_retrieve.return_value = []
    answer, sources = ask("pregunta")
    assert answer == "No encontré información relevante en los documentos."
    assert sources == []
