import streamlit as st
import requests

st.set_page_config(page_title="Legal RAG Chatbot", layout="wide")
st.title("⚖️ Asistente Legal Argentino")
st.markdown("Haz preguntas sobre leyes laborales argentinas.")

API_URL = "http://localhost:8000/ask"

question = st.text_area("Tu pregunta:", height=100)
if st.button("Consultar"):
    if question.strip():
        with st.spinner("Consultando..."):
            try:
                response = requests.post(API_URL, json={"text": question})
                if response.status_code == 200:
                    data = response.json()
                    st.subheader("Respuesta:")
                    st.write(data["answer"])
                    with st.expander("Ver fuentes"):
                        for i, src in enumerate(data["sources"]):
                            st.write(f"**Fuente {i+1}:** {src[:300]}...")
                else:
                    st.error("Error en la API")
            except Exception as e:
                st.error(f"Error de conexión: {e}")
    else:
        st.warning("Ingresa una pregunta")
