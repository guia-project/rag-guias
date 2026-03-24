
"""
Módulo de Lógica de Negocio y Motor RAG (Retrieval-Augmented Generation).

Este script actúa como el núcleo del sistema, gestionando la comunicación 
entre la base de datos vectorial (Elasticsearch), el modelo de embeddings 
y los diferentes proveedores de Modelos de Lenguaje (LLM).
"""
# Librerías
import json
import time
import requests
import warnings
from abc import ABC, abstractmethod
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from mistralai.client import Mistral
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings

#############################
## 0. CONFIGURACIÓN GLOBAL ##
#############################

try:
    with open('config.json', 'r') as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    print("ERROR: No se encuentra el archivo config.json.")
    exit()
# Variables de entorno extraídas de la configuración centralizada.
INDEX_NAME = CONFIG["elastic"]["index_name"]
MODEL_NAME = CONFIG["embeddings"]["model_name"]
RERANKER_MODEL = CrossEncoder(CONFIG["reranker"]["model_name"])

#######################################
## 1. DEFINICIÓN INTERFAZ ABSTRACTA ##
#######################################

class LLMProvider(ABC):
    """
    Clase base abstracta para los proveedores de LLM.
    
    Define el contrato que todas las implementaciones deben seguir.
    """
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Genera una respuesta basada en un prompt dado.
        
        Args:
            prompt (str): El texto de entrada para el modelo.
            
        Returns:
            str: La respuesta generada por el modelo.
        """
        pass

#########################################
## 2. IMPLEMENTACIONES DE PROVEEDORES  ##
#########################################

class MistralProvider(LLMProvider):
    """Implementación del proveedor Mistral AI."""

    def __init__(self, api_key, model_name):
        """
        Constructor de la clase MistralProvider.

        Args:
            api_key (str): Clave de API para autenticación en Mistral AI.
            model_name (str): Identificador del modelo.
        """
        self.api_key = api_key
        self.model_name = model_name
        self.client = Mistral(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        """
        Envía el prompt a la API de Mistral con instrucciones de sistema.

        Args:
            prompt (str): Texto que combina contexto y pregunta del usuario.

        Returns:
            str: Texto generado por el modelo o mensaje de error.
        """
        if not self.api_key:
            return "ERROR: Clave MISTRAL_API_KEY no configurada."
        max_retries = 3
        for i in range(max_retries):
            try:
                sys_instr = "Eres un Asistente de Guías Docentes. Responde basándote SOLO en el contexto."
                    
                response = self.client.chat.complete(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": sys_instr},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
                return response.choices[0].message.content
            except Exception as e:
                if "429" in str(e):
                    wait_time = (i + 1) * 20
                    time.sleep(wait_time)
                    continue
                return f"ERROR Mistral: {e}"
        return "ERROR Mistral: Máximo de reintentos alcanzado debido a limitaciones de tasa."

class GroqProvider(LLMProvider):
    """Implementación del proveedor Groq."""

    def __init__(self, api_key, model_name):
        """
        Constructor de la clase GroqProvider.

        Args:
            api_key (str): Clave de API de Groq.
            model_name (str): Identificador del modelo en Groq.
        """
        self.api_key = api_key
        self.model_name = model_name
        self.client = Groq(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        """
        Realiza una petición de chat completion a la API de Groq.

        Args:
            prompt (str): Entrada estructurada para el modelo.

        Returns:
            str: Contenido del mensaje de respuesta generado.
        """
        if not self.api_key:
            return "ERROR: Clave GROQ_API_KEY no configurada."
        max_retries = 3
        for i in range(max_retries):
            try:
                sys_instr = "Eres un Asistente de Guías Docentes. Responde basándote SOLO en el contexto."
                
                chat = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": sys_instr},
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model_name,
                    temperature=0.0
                )
                return chat.choices[0].message.content
            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    wait_time = (i + 1) * 20
                    time.sleep(wait_time)
                    continue
                return f"ERROR Groq: {e}"
        return "ERROR Groq: Máximo de reintentos alcanzado debido a limitaciones de tasa."

class OllamaProvider(LLMProvider):
    """Implementación del proveedor Ollama para ejecución local."""

    def __init__(self, api_url, model_name):
        """
        Constructor de la clase OllamaProvider.

        Args:
            api_url (str): URL del endpoint local de Ollama.
            model_name (str): Nombre del modelo descargado localmente.
        """
        self.api_url = api_url
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        """
        Envía una petición POST al servicio local de Ollama.

        Args:
            prompt (str): Texto de entrada.

        Returns:
            str: Respuesta generada por el modelo local.
        """
        max_retries = 3
        for i in range(max_retries):
            try:
                sys_instr = "Eres un Asistente de Guías Docentes. Responde basándote SOLO en el contexto.\n\n"
                full_prompt = sys_instr + prompt
                response = requests.post(
                    self.api_url,
                    json={
                        "model": self.model_name, 
                        "prompt": full_prompt, 
                        "stream": False,
                        "options": {"temperature": 0.0}
                    },
                    timeout=300
                )
                response.raise_for_status()
                return response.json()["response"]
            except Exception as e:
                if "429" in str(e) or "503" in str(e) or "Timeout" in str(type(e).__name__):
                    wait_time = (i + 1) * 5
                    print(f"Ollama ocupado o timeout. Reintentando en {wait_time}s... (Intento {i+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                return f"ERROR Ollama: {e}"
        return "ERROR Ollama: Máximo de reintentos alcanzado o servidor local no disponible."
#############################
## 3. FACTORÍA DE OBJETOS  ##
#############################

def get_llm_provider(force_provider_name=None) -> LLMProvider:
    """
    Lee la configuración y devuelve la instancia de la clase LLMProvider correcta.
    
    Args:
        force_provider_name (str, optional): Permite forzar un proveedor específico. 
            Si no se indica, se toma el valor 'active_llm' del config.json.

    Returns:
        LLMProvider: Una instancia concreta (Gemini, Groq u Ollama) según la elección.

    Raises:
        ValueError: Si el proveedor no existe en la configuración o es desconocido.
    """
    if force_provider_name:
        active_llm = force_provider_name
    else:
        active_llm = CONFIG["active_llm"]
    options = CONFIG["llm_options"].get(active_llm)

    if not options:
        raise ValueError(f"Configuración no encontrada para: {active_llm}")

    if active_llm == "mistral":
        return MistralProvider(
            api_key=options.get("api_key"),
            model_name=options["model"]
        )
    elif active_llm == "groq":
        return GroqProvider(
            api_key=options.get("api_key"),
            model_name=options["model"]
        )
    elif active_llm == "ollama":
        return OllamaProvider(
            api_url=options["api_url"],
            model_name=options["model"]
        )
    else:
        raise ValueError(f"Proveedor desconocido: {active_llm}")

###################################
## 4. CONEXIÓN Y RECUPERACIÓN    ##
###################################

def connect_to_elastic():
    """
    Inicializa la conexión con Elasticsearch.
    
    Returns:
        Elasticsearch: Cliente conectado.
        None: Si falla la conexión.
    """
    print(f"Conectando a Elasticsearch...")
    conf = CONFIG["elastic"]
    try:
        warnings.filterwarnings("ignore", "Connecting to",)
        client = Elasticsearch(
            [{"host": conf["host"], "port": conf["port"], "scheme": conf["scheme"]}],
            verify_certs=False, 
            request_timeout=10 
        )
        if client.ping():
            print("¡Conexión con Elasticsearch exitosa!")
            return client
    except Exception as e:
        print(f"Error conectando a Elasticsearch: {e}")
        return None

def load_embedding_model():
    """
    Carga el modelo de SentenceTransformer para generar representaciones vectoriales.

    Returns:
        SentenceTransformer: El modelo cargado en memoria listo para codificar texto.
        None: Si ocurre un error durante la carga.
    """
    print(f"Cargando modelo de embedding...")
    try:
        return HuggingFaceEmbeddings(model_name=MODEL_NAME)
    except Exception as e:
        print(f"Error al cargar modelo: {e}")
        return None

def search_retriever(client, model, query_text, top_k=10):
    """
    Ejecuta un proceso de recuperación avanzado en dos pasos: 
    1) Búsqueda Híbrida (combinando k-NN vectorial y coincidencia de texto BM25) para obtener candidatos 
    2) Reranking mediante un modelo Cross-Encoder para reordenar los resultados por su relevancia semántica 
    específica respecto a la consulta.

    Args:
        client (Elasticsearch): Cliente de la base de datos.
        model (SentenceTransformer): Modelo para vectorizar la consulta.
        query_text (str): La pregunta del usuario.
        top_k (int): Número de fragmentos a recuperar.

    Returns:
        tuple[list[str], list[str]]: Tupla que contiene una lista de contextos padres (secciones completas) 
            reordenados por relevancia y una lista de URLs de origen únicas.
        """
    try:
        query_vector = model.embed_query(query_text)
        search_query = {
            "size": 100,
            "query": {
                "bool": {
                    "should": [
                        { "match": { "chunk_text": { "query": query_text, "boost": 1.0 } } },
                        { "match_phrase": { "chunk_text": { "query": query_text, "boost": 5.0 } } },
                        { "knn": { 
                            "field": "embedding_vector", 
                            "query_vector": query_vector,  
                            "num_candidates": 200, 
                            "boost": 2.0 
                        } }
                    ],
                    "minimum_should_match": 1
                }
            },
            # Recuperamos el contexto padre (Mejora Parent-Document)
            "_source": ["chunk_text", "parent_context", "document_url"]
        }
        response = client.search(index=INDEX_NAME, body=search_query)
        hits = response['hits']['hits']
        if not hits: return [], []

        # Mejora Reranking
        documents = [hit['_source'] for hit in hits]
        pairs = [[query_text, doc['chunk_text']] for doc in documents]
        scores = RERANKER_MODEL.predict(pairs)

        for i, score in enumerate(scores):
            documents[i]['rerank_score'] = score
            
        # Ordenamos por la puntuación del reranker
        ranked_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)

        # Para no saturar la API
        MAX_CHARS = 10000
        current_chars = 0
        final_contexts = []
        final_sources = []
        seen_contexts = set() # Para evitar contextos duplicados

        for doc in ranked_docs:
            parent = doc['parent_context']
            child = doc['chunk_text']
            if parent in seen_contexts:
                continue

            # Caso 1: Cabe la sección completa (Parent)
            if current_chars + len(parent) < MAX_CHARS:
                final_contexts.append(parent)
                seen_contexts.add(parent)
                current_chars += len(parent)
                final_sources.append(doc['document_url'])
            
            # Caso 2: No cabe la sección, pero cabe el fragmento corto (Child)
            elif current_chars + len(child) < MAX_CHARS:
                final_contexts.append(child)
                current_chars += len(child)
                final_sources.append(doc['document_url'])
                # Si metemos el corto, paramos aquí para no saturar
                break
            
            # Caso 3: Ya no cabe nada más
            else:
                break

            if len(final_contexts) >= top_k:
                break

        return final_contexts, list(set(final_sources))

    except Exception as e:
        print(f"Error búsqueda: {e}")
        return [], []

def build_rag_prompt(query, context_chunks):
    """
    Construye un prompt estructurado inyectando el contexto extendido (Parent Context) de las secciones recuperadas. 
    Instruye al modelo para interpretar etiquetas jerárquicas y mitigar alucinaciones mediante el uso estricto de la información proporcionada.
    Args:
        query (str): La pregunta original del usuario.
        context_chunks (list[str]): Lista de Parent Documents (secciones enriquecidas con jerarquía) obtenidos tras el proceso de Reranking.
    Returns:
        str: Prompt estructurado listo para ser enviado al LLM.
    """
    context = "\n\n".join(context_chunks)
    return f"""
    Eres el Asistente Oficial de Guías Docentes de la UPM. Tu misión es responder de forma técnica y unificada.

    INSTRUCCIONES DE PROCESAMIENTO:
    1. El contexto está dividido en bloques. Cada bloque indica primero a qué asignatura pertenece y luego su ruta jerárquica. Ejemplo: [Asignatura: Nombre de la Asignatura] [X. Sección > X.Y. Subsección].
    2. Usa estas etiquetas para asegurar que estás respondiendo sobre la asignatura correcta que pide el usuario.
    3. Si la información en los fragmentos es contradictoria o pertenece a secciones distintas, explica la diferencia al alumno.
    4. Prohibido inventar datos (alucinaciones). Si la respuesta no está en el contexto, di: "Lo siento, esa información específica no consta en la guía docente actual".

    REGLAS DE FORMATO:
    - Responde de forma estructurada (usa viñetas si hay varios requisitos).
    - Sé directo. No uses frases de relleno como "Basado en el contexto proporcionado...".
    - Si mencionas un porcentaje o fecha, indica a qué sección o convocatoria pertenece.

    CONTEXTO RECUPERADO:
    {context}

    PREGUNTA DEL ALUMNO: {query}
    RESPUESTA FINAL:
    """

########################
## 5. BUCLE PRINCIPAL ##
########################

if __name__ == "__main__":
    """
    Ejecución del asistente en modo consola.
    """
    es_client = connect_to_elastic()
    embedding_model = load_embedding_model()
    
    try:
        llm_engine = get_llm_provider()
        print(f"Model cargado: {type(llm_engine).__name__}")
    except Exception as e:
        print(f"Error al configurar LLM: {e}")
        exit()

    if not es_client or not embedding_model:
        exit()

    print("\n" + "="*50)
    print(f"   Asistente RAG (LLM Activo: {CONFIG['active_llm']})")    
    print("="*50)

    try:
        while True:
            user_query = input("\n[Pregunta]: ")
            if user_query.lower() in ['salir', 'exit']: break
            
            print("... recuperando contexto ...")
            chunks, sources = search_retriever(es_client, embedding_model, user_query, top_k=10)
            
            if not chunks:
                print("No se encontró información relevante.")
                continue

            print("... generando respuesta ...")
            prompt = build_rag_prompt(user_query, chunks)
            answer = llm_engine.generate(prompt)
            
            print("\n[Respuesta]:")
            print(answer)
            
            if sources:
                print("\nFuentes:")
                for url in sources: print(f"- {url}")

    except KeyboardInterrupt:
        print("\nCerrando...")