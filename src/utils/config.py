import os
from dotenv import load_dotenv

load_dotenv()

def get_openai_api_key():
    return os.getenv("OPENAI_API_KEY")

def get_pinecone_api_key():
    return os.getenv("PINECONE_API_KEY")

def get_pinecone_environment():
    return os.getenv("PINECONE_ENVIRONMENT")

def get_pinecone_index_name():
    return os.getenv("PINECONE_INDEX_NAME", "legal-chatbot")

def get_s3_bucket():
    return os.getenv("S3_BUCKET_NAME")