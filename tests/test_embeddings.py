from unittest.mock import patch, MagicMock
from src.ingestion.embeddings_and_pinecone import embed_and_upload


# Helper para simular el comportamiento de fastembed (generador de arrays)
class MockEmbeddingResult:
    def __init__(self, data):
        self.data = data

    def tolist(self):
        return self.data


@patch("src.ingestion.embeddings_and_pinecone.pinecone")
@patch("src.ingestion.embeddings_and_pinecone.embedding_model")
@patch("src.ingestion.embeddings_and_pinecone.get_pinecone_api_key")
@patch("src.ingestion.embeddings_and_pinecone.get_pinecone_environment")
@patch("src.ingestion.embeddings_and_pinecone.get_pinecone_index_name")
def test_embed_and_upload(
    mock_index_name, mock_env, mock_key, mock_embed_model, mock_pinecone
):
    mock_key.return_value = "fake-key"
    mock_env.return_value = "gcp-starter"
    mock_index_name.return_value = "test-index"
    mock_pinecone.Pinecone.return_value.list_indexes.return_value.names.return_value = []
    mock_index = MagicMock()
    mock_pinecone.Pinecone.return_value.Index.return_value = mock_index

    class DummyChunk:
        def __init__(self, content):
            self.page_content = content
            self.metadata = {"source": "test"}

    chunks = [DummyChunk("texto1"), DummyChunk("texto2")]

    # Simular fastembed: devuelve un generador de objetos con .tolist()
    def mock_embed(texts):
        for _ in texts:
            yield MockEmbeddingResult([0.1] * 384)

    mock_embed_model.embed.side_effect = mock_embed

    embed_and_upload(chunks, batch_size=1)

    mock_pinecone.Pinecone.return_value.create_index.assert_called_once()
    assert mock_index.upsert.call_count == 2
