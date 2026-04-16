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
    load_reranker_model,
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

st.markdown("""
<style>
    /* Transición para logo y info. personal */
    .top-left-logo, .footer-left {
        transition: transform 0.3s ease-in-out, margin-left 0.3s ease-in-out;
    }

    /* Si la sidebar está EXPANDIDA (aria-expanded="true"), movemos los elementos a la derecha */
    [data-testid="stAppViewContainer"]:has([data-testid="stSidebar"][aria-expanded="true"]) .top-left-logo {
        margin-left: 336px;
    }
    
    [data-testid="stAppViewContainer"]:has([data-testid="stSidebar"][aria-expanded="true"]) .footer-left {
        margin-left: 336px;
    }

    .top-left-logo {
        position: fixed;
        top: 15px;
        left: 30px;
        z-index: 1000000; 
    }
    
    /* Fondo de la barra lateral en Azul UPM */
    [data-testid="stSidebar"] {
        background-color: #0072CE;
    }

    /* Texto de la barra lateral en Blanco */
    [data-testid="stSidebar"] .stText, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] p {
        color: white !important;
    }

    /* Estilo del título principal */
    .main .block-container h1 {
        color: #0072CE;
    }
    
    /* Mejorar visibilidad del selector de modelos en la sidebar */
    div[data-baseweb="select"] > div {
        background-color: white;
        color: black;
    }
</style>

<style>
    /* Estilo del botón flotante */
    .floating-about {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background-color: #0072CE;
        color: white;
        padding: 10px 20px;
        border-radius: 25px; /* Bordes redondeados */
        font-family: sans-serif;
        font-size: 14px;
        font-weight: bold;
        cursor: pointer;
        z-index: 9999; /* Asegura que esté por encima de todo */
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
        transition: background-color 0.3s;
    }

    /* Efecto al pasar el ratón por el botón */
    .floating-about:hover {
        background-color: #00569B; /* Un azul un poco más oscuro */
    }

    /* Estilo de la caja de texto oculta */
    .about-tooltip {
        visibility: hidden;
        width: 250px;
        background-color: #FFFFFF;
        color: #262730;
        text-align: left;
        border-radius: 8px;
        padding: 15px;
        position: absolute;
        z-index: 1;
        bottom: 130%; /* Lo posiciona justo encima del botón */
        right: 0;
        opacity: 0;
        transition: opacity 0.3s;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.15);
        border: 1px solid #E0E0E0;
        font-weight: normal;
    }

    /* Muestra la caja de texto al pasar el ratón */
    .floating-about:hover .about-tooltip {
        visibility: visible;
        opacity: 1;
    }

    /* Estilo de la info.personal */        
    .footer-left {
        position: fixed;
        bottom: 20px;
        left: 20px;
        font-family: sans-serif;
        font-size: 12px; /* Un pelín más grande para que el logo respire */
        color: #FFFFFF;
        z-index: 9999;
        background-color: #0072CE; 
        padding: 6px 12px;
        border-radius: 10px;
        
        /* Flexbox para alinear el texto "Hecho por" con el enlace */
        display: flex;
        align-items: center;
        gap: 6px; 
        
        transition: transform 0.3s ease-in-out, margin-left 0.3s ease-in-out;
    }
    
    /* Estilo específico para el enlace para alinear el logo y el nombre */
    .footer-left a {
        color: white !important;
        text-decoration: none;
        font-weight: bold;
        display: flex;
        align-items: center;
        gap: 5px; /* Espacio entre el logo y tu nombre */
    }
    
    .footer-left a:hover {
        text-decoration: underline;
    }
    
    /* Para que se mueva con la sidebar (lo que ya teníamos) */
    [data-testid="stAppViewContainer"]:has([data-testid="stSidebar"][aria-expanded="true"]) .footer-left {
        margin-left: 336px;
    }
            
</style>
            
<div class="top-left-logo">
    <img src="https://www.upm.es/gsfs/SFS46894" width="280">
</div>
            
<div class="footer-left">
    Hecho por 
    <a href="https://www.linkedin.com/in/gonzalo-sanz-sánchez-129a8731b" target="_blank">
        Gonzalo Sanz Sánchez
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white" width="14px" height="14px">
            <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
        </svg>
    </a>
</div>

<div class="floating-about">
    About
    <div class="about-tooltip">
        <b>Asistente de Guías Docentes UPM</b><br><br>
        Esta herramienta nace como solución a la imposibilidad de hacer consultas, pues las guías están en PDFs estáticos.<br><br>
        <i>Versión 1.0</i>
    </div>
</div>
""", 
unsafe_allow_html=True)

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
    reranker_model = load_reranker_model()
    return es_client, embedding_model, reranker_model


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
es_client, embedding_model, reranker_model = load_infrastructure()
llm_engine = load_llm_engine(selected_llm)

# Verificación de integridad del sistema
if not es_client or not embedding_model or not llm_engine or not reranker_model:
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
        if message.get("sources"):
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
        try:
            with st.spinner('Optimizando consulta...'):
                rewrite_prompt = "Extrae ÚNICAMENTE las palabras clave de esta pregunta (entidades, nombre de la asignatura, conceptos). NO uses frases completas, comillas, ni signos de puntuación. Separa las palabras por espacios."
                rewritten_query = llm_engine.generate(prompt, system_prompt=rewrite_prompt).strip()
                
            with st.spinner('Buscando en las guías docentes...'):    
                # 1. Recuperación (Retrieve)
                chunks, sources = search_retriever(
                    es_client, 
                    embedding_model, 
                    reranker_model,
                    prompt,
                    rewritten_query, 
                    top_k=10 
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