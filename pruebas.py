"""
Módulo de Evaluación de Calidad de Respuestas (TFG).

Este script automatiza la evaluación de un sistema RAG utilizando métricas 
de procesamiento de lenguaje natural (NLP): BLEU, METEOR y ROUGE.
"""

# Librerías
import json
import evaluate
from app import connect_to_elastic, load_embedding_model, get_llm_provider, search_retriever, build_rag_prompt

##########################
## 1. CARGA DE MÉTRICAS ##
##########################

bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")

def run_quality_test(dataset_path):
    """
    Ejecuta el pipeline RAG sobre un dataset y calcula métricas de calidad.
    
    Args:
        dataset_path (str): Ruta al archivo JSON con preguntas y referencias.
    """
    # Inicialización de componentes.
    es_client = connect_to_elastic()
    embed_model = load_embedding_model()
    llm_engine = get_llm_provider()
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    predictions = []
    references = []

    print(f"Iniciando evaluación sobre {len(dataset)} preguntas...")

    for item in dataset:
        pregunta = item["pregunta"]
        verdad_humana = item["referencia"]
        
        # Flujo RAG (Retrieval + Generation)
        chunks, _ = search_retriever(es_client, embed_model, pregunta, top_k=5)
        prompt = build_rag_prompt(pregunta, chunks)
        respuesta_ia = llm_engine.generate(prompt)
        
        predictions.append(respuesta_ia)
        references.append(verdad_humana)
        print(f"Pregunta procesada: {pregunta[:50]}...")
    
    ############################
    ## 2. CÁLCULO DE MÉTRICAS ##
    ############################

    # BLEU: Mide precisión de n-gramas (común en traducción)
    results_bleu = bleu.compute(predictions=predictions, references=references)
    
    # METEOR: Considera sinónimos y morfología (más alineado con el lenguaje humano)
    results_meteor = meteor.compute(predictions=predictions, references=references)
    
    # ROUGE: Mide el 'recall' (cuánta información de la referencia está en la IA)
    results_rouge = rouge.compute(predictions=predictions, references=references)

    ###################
    ## 3. RESULTADOS ##
    ###################

    print("\n" + "="*30)
    print("RESULTADOS DE EVALUACIÓN")
    print("="*30)
    print(f"BLEU Score: {results_bleu['bleu']:.4f}")
    print(f"METEOR Score: {results_meteor['meteor']:.4f}")
    print(f"ROUGE-L: {results_rouge['rougeL']:.4f}")

if __name__ == "__main__":
    run_quality_test("eval_dataset.json")