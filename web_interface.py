# Módulo de Interfaz Web para el Asistente de Guías Docentes de la UPM.
# 
# Este script utiliza Streamlit para proporcionar una interfaz de chat amigable
# que permite a los usuarios consultar información sobre asignaturas. Implementa
# estrategias de caché para optimizar la carga de modelos y la conexión a bases de datos.

# Librerías
import streamlit as st

# Importamos funciones y clases desde app.py para mantener la lógica de negocio separada de la interfaz.
from app import (
    connect_to_elastic, 
    load_embedding_model, 
    get_llm_provider, 
    search_retriever, 
    build_rag_prompt,
    CONFIG
)

#####################################
##  1. CONFIGURACIÓN DE LA PÁGINA  ##
#####################################

# st.set_page_config para definir el título, icono y diseño de la página.
st.set_page_config(
    page_title="Asistente Guías Docentes",
    page_icon="🎓",
    layout="centered"
)
# st.title para mostrar un título llamativo en la parte superior de la página.
st.title("🎓 Chatbot Guías Docentes UPM")

# Configuración de la barra lateral.    
# Permite al usuario seleccionar dinámicamente el proveedor de LLM 
# basándose en las opciones disponibles en el archivo de configuración,
# siempre que la variable de visibilidad esté activa.
with st.sidebar:
    st.header("Configuración")


    show_llm_selector = CONFIG.get("show_llm_selector")
    
    available_models = list(CONFIG["llm_options"].keys())

    if show_llm_selector:
        selected_llm = st.selectbox(
            "Selecciona el modelo LLM:",
            options=available_models,
            index=available_models.index(CONFIG["active_llm"]))
    else:
        selected_llm = CONFIG["active_llm"]
        st.info(f"Modelo activo: {selected_llm}")


##################################
## 2. CARGA DE RECURSOS (CACHÉ) ##
##################################

@st.cache_resource
def load_infrastructure():
    # Inicializa y cachea la infraestructura base.
    # 
    # Establece la conexión con Elasticsearch y carga el modelo de embeddings
    # en memoria para evitar recargas costosas en cada interacción de Streamlit.
    # 
    # Returns:
    #     tuple: (es_client, embedding_model) donde es_client es la instancia de 
    #            Elasticsearch y embedding_model es el modelo SentenceTransformer.
    es_client = connect_to_elastic()
    embedding_model = load_embedding_model()
    return es_client, embedding_model


@st.cache_resource
def load_llm_engine(provider_name):
    # Carga y cachea el motor del Modelo de Lenguaje (LLM).
    # 
    # Utiliza el patrón Factory para instanciar el proveedor correcto (Gemini, Groq, etc.)
    # y mantiene la instancia en caché para agilizar la generación de respuestas.
    # 
    # Args:
    #     provider_name (str): Nombre del proveedor de LLM (ej. 'ollama', 'groq').
    # 
    # Returns:
    #     LLMProvider: Una instancia de la clase proveedora de LLM configurada.
    #     None: Si ocurre un error durante la inicialización.
    try:
        return get_llm_provider(force_provider_name=provider_name)
    except Exception as e:
        st.error(f"Error al cargar {provider_name}: {e}")
        return None

# Inicialización de recursos globales
es_client, embedding_model = load_infrastructure()
llm_engine = load_llm_engine(selected_llm)

# Verificación de integridad del sistema
if not es_client or not embedding_model or not llm_engine:
    st.error("Error crítico de conexión. Revisa la terminal.")
    st.stop()
##############################################
## 3. GESTIÓN DEL HISTORIAL (SESSION STATE) ##
##############################################

if "messages" not in st.session_state:
    # Inicializa el historial de conversación en el estado de la sesión
    # si es la primera vez que se carga la aplicación.
    st.session_state.messages = []

# Renderizamos el historial de mensajes almacenado en la sesión para mostrar la conversación previa.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Fuentes consultadas"):
                for source in message["sources"]:
                    st.markdown(f"- [{source}]({source})")

########################
## 4. LÓGICA DEL CHAT ##
########################

if prompt := st.chat_input("Pregunta sobre una asignatura..."):
    # Bucle principal de procesamiento de preguntas (Pipeline RAG).
    # 
    # 1. Captura la entrada del usuario.
    # 2. Recupera fragmentos relevantes (Retrieval) de Elasticsearch.
    # 3. Construye el prompt con contexto.
    # 4. Genera la respuesta definitiva (Generation) a través del LLM.
    # 5. Actualiza el historial de la sesión.
    
    # Muestra mensaje del usuario.
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Genera respuesta del asistente.
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

                # Renderizar la respuesta del asistente y fuentes.
                message_placeholder.markdown(full_response)
                
                if sources:
                    with st.expander("Fuentes consultadas"):
                        for url in sources:
                            st.markdown(f"- [{url}]({url})")

                # Actualizar el historial de la sesión con la respuesta del asistente y las fuentes.
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Ocurrió un error: {e}")