import pinecone
from fastembed import TextEmbedding
from src.utils.config import get_pinecone_api_key, get_pinecone_environment, get_pinecone_index_name
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")  # dimensión 384

def initialize_pinecone():
    pc = pinecone.Pinecone(
        api_key=get_pinecone_api_key(),
        environment=get_pinecone_environment()
    )
    index_name = get_pinecone_index_name()
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=pinecone.ServerlessSpec(
                cloud="aws",
                region=get_pinecone_environment()
            )
        )
        logger.info(f"Índice '{index_name}' creado.")
    else:
        logger.info(f"Índice '{index_name}' ya existe.")
    return pc.Index(index_name)

def embed_and_upload(chunks, batch_size=100):
    index = initialize_pinecone()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [chunk.page_content for chunk in batch]
        metadatas = [chunk.metadata for chunk in batch]
        ids = [f"chunk_{i+j}" for j in range(len(batch))]
        
        vectors = list(embedding_model.embed(texts))
        vectors = [v.tolist() for v in vectors]
        
        to_upsert = [(ids[k], vectors[k], metadatas[k]) for k in range(len(batch))]
        index.upsert(vectors=to_upsert)
        logger.info(f"Subidos {len(batch)} chunks a Pinecone")
    logger.info("Ingesta completa.")