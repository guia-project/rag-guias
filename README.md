# Asistente RAG para Guías Docentes

Este proyecto implemtenta un sistema de **Generación Aumentada por Recuperación (RAG)** avanzado para la consulta de guías docentes de la Universidad Politécnica de Madrid. El sistema utiliza una arquitectura de microservicios para gestionar la ingesta de datos, el almacenamiento vectorial y la interfaz de usuario.

## Características Principales
* **Ingesta Automática**: Sincronización con el repositorio CKAN para la ingesta periódica de datos.
* **Procesamiento Avanzado**: Pipeline de "Triple Chunking" (Semántico, Markdown y Recursivo).
* **Búsqueda Híbrida**: Combinación de búsqueda vectorial (semántica) y léxica (palabras clave).
* **Reranking**: Re-clasificación de resultados mediante modelos Cross-Encoder para máxima precisión.

* **Variedad de LLMs**: Soporte para modelos locales (Ollama) y en la nube (Mistral, Groq).

* **Evaluación Científica**: Módulo integrado para cálculo de métricas NLP (BLEU, METEOR, ROUGE-L).

## Estructura del Prroyecto

El sistema sigue una arquitectura de diseño modular para facilitar el mantenimiento y la escalabilidad:

    rag-guias/
    ├── .streamlit/             # Configuración  de Streamlit (config.toml)
    ├── config/                 # Configuración centralizada (JSON)
    │   ├── config.json         # Fichero de configuración
    │   ├── example_config.json # Fihcero de configuración de ejemplo
    ├── data/                   # Datasets de evaluación y resultados (CSV)
    ├── docker/                 # Definición de infraestructura y contenedores
    │   └── Dockerfile          # Receta de construcción de la imagen Python
    ├── src/                    # Lógica de negocio (Scripts de Python)
    │   ├── app.py              # Motor RAG y Proveedores de LLM
    │   ├── indexer.py          # Ingesta y Sincronización de datos
    │   ├── web_interface.py    # Interfaz de usuario (Streamlit)
    │   └── pruebas.py          # Banco de pruebas de calidad
    ├── .gitignore              # Archivos excluidos del control de versiones
    ├── docker-compose.yml      # Orquestación de microservicios
    ├── README.md               # Guía de uso y documentación técnica principal
    └── requirements.txt        # Dependencias del sistema

## Instalación y Despliegue
**Requisitos previos**
* Docker y Docker Compose
* Al menos 8GB de RAM disponibles (Se recomienda cerrar aplicaciones pesadas durante la ejecución)

**Puesta en marcha**
1. **Clonar el repositorio**
    
        git clone https://github.com/guia-project/rag-guias.git
        cd rag-guias
2. **Configurar credenciales**

    Crea en la carpeta `config/` el archivo `config/config.json` con tus API Keys si vas a usar groq o mistral.
3. **Lanzar el sistema**

        docker compose up -d --build
    El asistente estará disponible en: http://localhost:8501

## Evaluación de Calidad
Para ejecutar el banco de pruebas y generar el informe de métricas sobre el dataset de validación:

    docker exec -it rag_web_chatbot python pruebas.py
Los resultados se exportarán automáticamente a `data/eval_results.csv` , incluyendo comparativas de similitud entre las respuestas del modelo y las referencias humanas.

## Tecnologías Utilizadas
| Componente | Tecnoología |
| :-- | :-- |
| **Base de Datos** | Elasticsearch |
| **Orquestación** | Docker / Docker Compose|
| **Modelos Embedding** | HuggingFace (sentence-transformers)|
| **Modelos Reranking** | Cross-Encoder (MS MARCO)|
| **Backend / UI** | Python 3.11 / Streamlit|
| **LLMs** | Groq, Mistral, Ollama|
