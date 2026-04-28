"""
Módulo de Indexación y Sincronización de Guías Docentes.

Este script se encarga de la ingesta de datos desde el portal CKAN de la UPM,
su procesamiento (conversión a Markdown y fragmentación) y su posterior
almacenamiento vectorial en Elasticsearch para habilitar la búsqueda semántica.
"""
# Librerías
import requests
import schedule
import time
import io
import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from markitdown import MarkItDown
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

#############################
## 0. CONFIGURACIÓN GLOBAL ##
#############################
try:
    with open('config.json', 'r') as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    print("ERROR: No se encuentra el archivo config.json.")
    exit()

# Asignación de variables desde el archivo de configuración centralizado.
ELASTIC_CONFIG = CONFIG["elastic"]
CKAN_API_URL = CONFIG["ckan"]["api_url"]
MODEL_NAME = CONFIG["embeddings"]["model_name"]
EMBEDDING_DIM = CONFIG["embeddings"]["dim"]
INDEX_NAME = ELASTIC_CONFIG["index_name"]
CKAN_API_QUERY = CONFIG["ckan"]["query"]

#################################
## 1. FUNCIONES ELASTICSEARCH  ##
#################################

def connect_to_elastic():
    """
    Establece la conexión con el servidor de Elasticsearch.

    Utiliza los parámetros definidos en ELASTIC_CONFIG para inicializar el cliente.

    Returns:
        Elasticsearch: Cliente instanciado si la conexión es exitosa.
        None: Si ocurre un error durante la conexión.
    """
    print(f"Conectando a Elasticsearch...")
    try:
        client = Elasticsearch(
            [{"host": ELASTIC_CONFIG["host"], "port": ELASTIC_CONFIG["port"], "scheme": ELASTIC_CONFIG["scheme"]}],
            verify_certs=False,
            request_timeout=60,
            max_retries = 3,
            retry_on_timeout=False
        )
        info = client.info()
        print("Conexión con Elasticsearch exitosa")
        return client
    except Exception as e:
        print(f"Error conectando a Elasticsearch: {e}")
        return None

def create_index_mapping(client):
    """
    Crea el índice en Elasticsearch con un mapeo específico para búsqueda vectorial.

    Define campos para metadatos (IDs, URLs) y el campo 'embedding_vector' 
    de tipo 'dense_vector' para almacenar los vectores generados por el modelo de HuggingFace.

    Args:
        client (Elasticsearch): El cliente de conexión activo.
    """
    if client.indices.exists(index=INDEX_NAME):
        print(f"El índice '{INDEX_NAME}' ya existe.")
        return

    mapping_body = {
        "mappings": {
            "properties": {
                "document_id": { "type": "keyword" }, 
                
                "document_url": { "type": "keyword" },
                
                "chunk_text": { "type": "text" }, 

                "parent_context": { "type": "text" },
                
                "embedding_vector": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIM 
                },
                "modified_date": { "type": "date" } 
            }
        }
    }
    try:
        client.indices.create(index=INDEX_NAME, body=mapping_body)
        print(f"Índice '{INDEX_NAME}' creado con éxito.")
    except Exception as e:
        print(f"Error al crear el índice: {e}")

################################
## 2. LÓGICA DE PROCESAMIENTO ##
################################

def convert_pdf_to_markdown(pdf_content_bytes):
    """
    Transforma el contenido binario de un PDF a formato Markdown.

    Utiliza la librería MarkItDown para extraer el texto preservando 
    la estructura semántica básica del documento.

    Args:
        pdf_content_bytes (bytes): Contenido del archivo PDF en crudo.

    Returns:
        str: Texto convertido a Markdown.
        None: Si la conversión falla.
    """
    md_converter = MarkItDown()
    try:
        with io.BytesIO(pdf_content_bytes) as f:
            result = md_converter.convert_stream(f)
        return result.text_content
    except Exception as e:
        print(f"Error al convertir PDF con MarkItDown: {e}")
        return None

def get_chunks_from_markdown(markdown_content, semantic_chunker, recursive_chunker, doc_name):
    """
    Divide un texto largo en fragmentos (chunks) estructurados mediante una estrategia de triple fase: 
    1) División estructural por cabeceras Markdown, 
    2) División semántica por coherencia temática mediante embeddings, 
    3) Refinamiento recursivo para garantizar que no se exceda el límite de tokens del modelo.

    Args:
        markdown_content (str): Contenido en Markdown
        semantic_chunker (SemanticChunker): Instancia para división temática.
        recursive_chunker (RecursiveCharacterTextSplitter): Instancia para refinamiento por tamaño.
        doc_name (str): Nombre o título de la asignatura para inyectarlo como contexto en cada fragmento.

    Returns:
        list[dict]: Lista de diccionarios, donde cada uno contiene search_text (hijo para búsqueda) 
        y parent_context (padre para respuesta).
    """

    # Parent-document retrieval
    headers_to_split_on = [
        ("#", "Sección_Principal"),
        ("##", "Subseccion"),
        ("###", "Apartado")
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    sections = markdown_splitter.split_text(markdown_content)


    chunks = []
    for doc in sections:
        parent_text = doc.page_content
        context_path = " > ".join([f"{v}" for k, v in doc.metadata.items()])

        semantic_splits = semantic_chunker.split_text(parent_text)

        for s in semantic_splits:
            final_s_chunks = recursive_chunker.split_text(s)
            for sub_chunk in final_s_chunks:
                chunks.append({
                    "search_text": f"[Asignatura: {doc_name}] [{context_path}]\n{sub_chunk}",
                    "parent_context": f"[Asignatura: {doc_name}] [{context_path}]\n{parent_text}"
                })
    return chunks


#################################
## 3. LÓGICA DEL SINCRONIZADOR ##
#################################

def fetch_ckan_resources():
    """
    Recupera de forma paginada todos los recursos disponibles en CKAN.

    Utiliza un bucle iterativo con 'limit' y 'offset' para garantizar 
    la obtención de la totalidad de las guías sin saturar la API.

    Returns:
        list: Lista de diccionarios con la información de cada recurso.
    """
    all_resources = []
    limit = 500
    offset = 0
    print(f"Llamando a la API de CKAN: {CKAN_API_URL}")
    try:
        while True:
            params = {
                "query": CKAN_API_QUERY,
                "limit": limit,
                "offset": offset
            }
            response = requests.get(CKAN_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
             
            if not data.get("success"):
                break

            results = data["result"]["results"]
            if not results:
                break
            all_resources.extend(results)
            print(f"-> Recuperados {len(all_resources)} recursos...")
            
            offset += limit
        print(f"Total recursos PDF obtenidos hasta ahora: {len(all_resources)}")
        return all_resources
    except requests.RequestException as e:    
        print(f"La API de CKAN devolvió un error: {e}")
        return all_resources

    
def document_exists(client, document_id):
    """
    Verifica la existencia de un documento en el índice actual.

    Args:
        client (Elasticsearch): Cliente de conexión.
        document_id (str): ID único del recurso en CKAN.

    Returns:
        bool: True si el documento ya está indexado, False en caso contrario.
    """
    try:
        count = client.count(index=INDEX_NAME, body={
            "query": { "term": { "document_id": document_id } }
        })
        return count["count"] > 0
    except Exception:
        return False

def process_and_index_document(client, model, resource, semantic_chunker, recursive_chunker):
    """
    Ejecuta el pipeline RAG completo para un único documento.

    Pasos: Descarga -> Conversión (Markdown) -> Triple Chunking (Estructural, Semántico, Recursivo) -> Vectorización -> Indexación Bulk con arquitectura Parent-Child.

    Args:
        client (Elasticsearch): Cliente de base de datos.
        model (HuggingFaceEmbeddings): Modelo para vectorizar.
        resource (dict): Metadatos del recurso de CKAN.
        semantic_chunker (SemanticChunker): Objeto configurado para la segmentación semántica.
        recursive_chunker (RecursiveCharacterTextSplitter): Objeto configurado para la segmentación recursiva.
    """
    doc_id = resource['id']
    doc_url = resource['url']
    mod_date = resource['metadata_modified']
    
    doc_name = resource.get('name', resource.get('title', 'Guía Docente'))

    # Descarga del PDF.
    print(f"-> Descargando: {doc_name}, ({doc_url})")
    try:
        pdf_response = requests.get(doc_url, timeout=30)    
        pdf_response.raise_for_status()

        # Procesamiento del PDF (MarkItDown + Chunking)
        print("-> Procesando con MarkItDown...")
        markdown_text = convert_pdf_to_markdown(pdf_response.content)    
        if not markdown_text:
            print("ERROR: MarkItDown no devolvió contenido.")
            return 
        
        print("-> Dividiendo en Chunks (RCTS)...")
        chunks = get_chunks_from_markdown(markdown_text, semantic_chunker, recursive_chunker, doc_name)        
        if not chunks:
            print("ERROR: No se generaron chunks.")
            return

        search_texts = [item["search_text"] for item in chunks]

        # Vectorización de los chunks con el modelo de embedding.
        print(f"-> Vectorizando {len(search_texts)} chunks...")
        embeddings = model.embed_documents(search_texts)


        # Inserción masiva en Elasticsearch utilizando Bulk API
        actions = []
        for i, chunk in enumerate(chunks):
            action = {
                "_index": INDEX_NAME, 
                "_source": {
                    "document_id": doc_id,
                    "document_url": doc_url,
                    "chunk_text": chunk["search_text"],
                    "parent_context": chunk["parent_context"],
                    "embedding_vector": embeddings[i],
                    "modified_date": mod_date
                }
            }
            actions.append(action)

        if actions:
            bulk(client, actions, chunk_size=50, request_timeout=60) 
            print(f"¡Éxito! Documento {doc_id} indexado.")
    except requests.RequestException as e:
        print(f"ERROR al descargar {doc_url}: {e}")
        return

    
        
def run_sync_job(client, model, semantic_chunker, recursive_chunker):
    """
    Coordina el proceso de sincronización global.

    Obtiene recursos de CKAN y procesa únicamente aquellos que 
    no existan previamente en Elasticsearch.

    Args:
        client (Elasticsearch): Cliente de conexión.
        model (HuggingFaceEmbeddings): Modelo para generar embeddings.
        semantic_chunker (SemanticChunker): Herramienta de segmentación semántica inicializada.
        recursive_chunker (RecursiveCharacterTextSplitter): Herramienta de segmentación recursiva inicializada.
    """
    
    ckan_resources = fetch_ckan_resources()
    if not ckan_resources:
        print("No se obtuvieron recursos de CKAN. Finalizando trabajo.")
        return

    for res in ckan_resources:
        if res.get('mimetype') == 'application/pdf' and not document_exists(client, res['id']):
            process_and_index_document(client, model, res, semantic_chunker, recursive_chunker)

############################
## 4. EJECUCIÓN PRINCIPAL ##
############################

if __name__ == "__main__":
    """
    Punto de entrada del script.
    
    Inicializa los clientes, ejecuta una sincronización inmediata y 
    programa las ejecuciones futuras cada 24 horas.
    """
    es_client = connect_to_elastic()
    
    if es_client:
        create_index_mapping(es_client)

        print(f"\nCargando modelo de embedding ({MODEL_NAME}) en memoria...")
        embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        print("Modelo de embedding cargado y listo.")
        
        sem_chunker = SemanticChunker(embedding_model, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=85)
        rec_chunker = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120, separators=["\n\n", "\n", ". ", " ", ""])

        # Primera ejecución
        print("\nEjecutando la primera sincronización al inicio...")
        run_sync_job(es_client, embedding_model, sem_chunker, rec_chunker)
        
        # Programación de ejecuciones futuras cada 24 horas
        print("\nProgramando el trabajo para ejecutarse cada 24 horas...")
        schedule.every(24).hours.do(run_sync_job, client=es_client, model=embedding_model, semantic_chunker=sem_chunker, recursive_chunker=rec_chunker)
        
        print("El indexador está ahora en modo 'schedule'. Presiona Ctrl+C para salir.")
        while True:
            schedule.run_pending()
            time.sleep(60) 
    else:
        print("\nERROR: No se pudo conectar a Elasticsearch. Saliendo.")
