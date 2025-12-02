import streamlit as st
import time

from app import (
    connect_to_elastic, 
    load_embedding_model, 
    get_llm_provider, 
    search_retriever, 
    build_rag_prompt,
    CONFIG
)

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="Asistente Guías Docentes",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Chatbot Guías Docentes UPM")

with st.sidebar:
    st.header("Configuración")

    available_models = list(CONFIG["llm_options"].keys())

    selected_llm = st.selectbox(
        "Selecciona el modelo LLM:",
        options=available_models,
        index=available_models.index(CONFIG["active_llm"])
    )

# 2. CARGA DE RECURSOS (CACHÉ)
# @st.cache_resource para que se ejecute SOLO UNA VEZ

@st.cache_resource
def load_infrastructure():
    es_client = connect_to_elastic()
    embedding_model = load_embedding_model()
    return es_client, embedding_model

es_client, embedding_model = load_infrastructure()

# Carga del LLM (Dinámica según el selector)
def load_llm_engine(provider_name):
    try:
        return get_llm_provider(force_provider_name=provider_name)
    except Exception as e:
        st.error(f"Error al cargar {provider_name}: {e}")
        return None

# Cargamos el motor seleccionado en la barra lateral
llm_engine = load_llm_engine(selected_llm)

# Verificación de salud
if not es_client or not embedding_model or not llm_engine:
    st.error("Error crítico de conexión. Revisa la terminal.")
    st.stop()

# 3. GESTIÓN DEL HISTORIAL (SESSION STATE)
# Streamlit se reinicia con cada interacción, con 'session_state' para recordar la conversación.

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes anteriores al recargar la página
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Si el mensaje tenía fuentes guardadas, las mostramos
        if "sources" in message:
            with st.expander("Fuentes consultadas"):
                for source in message["sources"]:
                    st.markdown(f"- [{source}]({source})")

# 4. LÓGICA DEL CHAT

if prompt := st.chat_input("Pregunta sobre una asignatura..."):
    
    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Guardar en historial
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generar respuesta del asistente
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner('Buscando en las guías docentes...'):
            try:
                # 1. Recuperación (Retrieve)
                chunks, sources = search_retriever(
                    es_client, 
                    embedding_model, 
                    prompt, 
                    top_k=5
                )
                
                if not chunks:
                    full_response = "Lo siento, no he encontrado información relevante en las guías indexadas para responder a tu pregunta."
                    sources = []
                else:
                    # 2. Construcción del Prompt
                    rag_prompt = build_rag_prompt(prompt, chunks)
                    
                    # 3. Generación (Generate)
                    full_response = llm_engine.generate(rag_prompt)

                # Mostrar respuesta
                message_placeholder.markdown(full_response)
                
                # Mostrar fuentes si existen
                if sources:
                    with st.expander("Fuentes consultadas"):
                        for url in sources:
                            st.markdown(f"- [{url}]({url})")

                # Guardar respuesta y fuentes en historial
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Ocurrió un error: {e}")