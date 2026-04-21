import os
from dotenv import load_dotenv

load_dotenv()

def get_pinecone_api_key():
    return os.getenv("PINECONE_API_KEY")

def get_pinecone_environment():
    return os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")

def get_pinecone_index_name():
    return os.getenv("PINECONE_INDEX_NAME", "legal-rag")

def get_groq_api_key():
    return os.getenv("GROQ_API_KEY")
