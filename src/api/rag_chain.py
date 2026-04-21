import pinecone
from fastembed import TextEmbedding
from groq import Groq
from src.utils.config import get_pinecone_api_key, get_pinecone_environment, get_pinecone_index_name, get_groq_api_key
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
groq_client = Groq(api_key=get_groq_api_key())

def get_pinecone_index():
    pc = pinecone.Pinecone(api_key=get_pinecone_api_key(), environment=get_pinecone_environment())
    index_name = get_pinecone_index_name()
    return pc.Index(index_name)

def retrieve(query: str, top_k=3):
    query_embedding = list(embedding_model.embed([query]))[0].tolist()
    index = get_pinecone_index()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    contexts = []
    for match in results['matches']:
        contexts.append(match['metadata'].get('text', ''))
    return contexts

def generate_answer(query: str, contexts: list) -> str:
    context_text = "\n\n".join(contexts)
    prompt = f"""Eres un asistente legal argentino. Responde la pregunta basándote únicamente en el contexto proporcionado. Si no puedes responder, di que no tienes información suficiente.

Contexto:
{context_text}

Pregunta: {query}

Respuesta:"""
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error en Groq: {e}")
        return "Lo siento, no pude generar una respuesta en este momento."

def ask(question: str):
    contexts = retrieve(question)
    if not contexts:
        return "No encontré información relevante en los documentos.", []
    answer = generate_answer(question, contexts)
    return answer, contexts
