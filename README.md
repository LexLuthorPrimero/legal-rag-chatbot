[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io/)
[![CI](https://github.com/LexLuthorPrimero/legal-rag-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/LexLuthorPrimero/legal-rag-chatbot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Legal RAG Chatbot

Sistema de preguntas y respuestas sobre leyes laborales argentinas usando Retrieval-Augmented Generation (RAG).

## Arquitectura

- **Ingesta**: documentos → chunking → embeddings (FastEmbed) → Pinecone (vector DB)
- **Consulta**: pregunta → embedding → búsqueda en Pinecone → contexto + LLM (Groq Llama 3) → respuesta
- **API**: FastAPI (endpoint `/ask`)
- **Frontend**: Streamlit

## Demo

![Interfaz](images/abogado.png)
[Ver demostración del chatbot funcionando](https://youtu.be/M4lLOUbNJiQ)

## Características

- Extracción de documentos desde archivos `.txt`
- Embeddings locales con FastEmbed
- Almacenamiento vectorial en Pinecone
- Generación de respuestas con Groq (`llama-3.3-70b-versatile`)
- API autodocumentada (OpenAPI)
- Interfaz de usuario con Streamlit
- Tests unitarios + CI (GitHub Actions)
- Pre-commit hooks (ruff, mypy)

## Instalación y uso

```bash
git clone https://github.com/LexLuthorPrimero/legal-rag-chatbot.git
cd legal-rag-chatbot
python -m venv venv
source venv/bin/activate
pip install -e .

# (Opcional) cargar documentos en Pinecone
python scripts/run_ingestion.py

# Ejecutar API
python scripts/run_api_local.py

# En otra terminal, ejecutar frontend
streamlit run src/frontend/app.py


Accede a http://localhost:8501 y prueba preguntas como:

    "¿Cuántas horas diarias se puede trabajar?"

    "¿Qué dice la ley sobre horas extra?"

Estructura
text

legal-rag-chatbot/
├── src/
│   ├── api/              # FastAPI
│   ├── frontend/         # Streamlit
│   ├── ingestion/        # carga, chunking, embeddings
│   └── utils/            # config, logging
├── tests/
├── scripts/
├── data/raw/leyes/
├── .github/workflows/
├── requirements.txt
├── pyproject.toml
└── README.md

Tests
bash

pytest tests/ -v --cov=src

Licencia

MIT
Autor

Lucas (LexLuthorPrimero)
GitHub | LinkedIn