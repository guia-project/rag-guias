"""
Módulo de Evaluación de Calidad de Respuestas (TFG).

Este script automatiza la evaluación de un sistema RAG utilizando métricas 
de procesamiento de lenguaje natural (NLP): BLEU, METEOR y ROUGE.
"""

# Librerías
import json
import pandas as pd
import numpy as np
import evaluate
from app import CONFIG, connect_to_elastic, load_embedding_model, get_llm_provider, search_retriever, build_rag_prompt

##########################
## 1. CARGA DE MÉTRICAS ##
##########################

bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")

output_csv = CONFIG["eval"]["output_csv"]
dataset_path = CONFIG["eval"]["dataset_path"]

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

    results_list = []

    print(f"Iniciando evaluación sobre {len(dataset)} preguntas...")

    for i, item in enumerate(dataset):
        pregunta = item["pregunta"]
        referencia = item["referencia"]
        
        # Flujo RAG (Retrieval + Generation)
        chunks, _ = search_retriever(es_client, embed_model, pregunta, top_k=5)
        prompt = build_rag_prompt(pregunta, chunks)
        respuesta_ia = llm_engine.generate(prompt)
        
        # Cálculo de métricas individuales para esta pregunta
        # (Usamos listas de un solo elemento para evaluar pregunta por pregunta)
        score_bleu = bleu.compute(predictions=[respuesta_ia], references=[referencia])['bleu']
        score_meteor = meteor.compute(predictions=[respuesta_ia], references=[referencia])['meteor']
        score_rouge = rouge.compute(predictions=[respuesta_ia], references=[referencia])['rougeL']
        
        # Guardamos los datos de esta iteración
        results_list.append({
            "ID": i + 1,
            "Pregunta": pregunta,
            "Respuesta_IA": respuesta_ia,
            "Referencia_Humana": referencia,
            "BLEU": score_bleu,
            "METEOR": score_meteor,
            "ROUGE-L": score_rouge
        })
        
        print(f"[{i+1}/{len(dataset)}] Procesada: {pregunta[:40]}...")
    
    ############################
    ## 2. CÁLCULO DE MÉTRICAS ##
    ############################

    df = pd.DataFrame(results_list)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    # Calculamos medias y desviaciones típicas
    summary = {
        "Métrica": ["BLEU", "METEOR", "ROUGE-L"],
        "Media (μ)": [df["BLEU"].mean(), df["METEOR"].mean(), df["ROUGE-L"].mean()],
        "Desv. Típica (σ)": [df["BLEU"].std(), df["METEOR"].std(), df["ROUGE-L"].std()]
    }
    df_summary = pd.DataFrame(summary)

    ###################
    ## 3. RESULTADOS ##
    ###################

    print("\n" + "="*45)
    print("RESULTADOS DE EVALUACIÓN")
    print("="*45)
    print(df_summary.to_string(index=False))
    print("="*45)
    print(f"Fichero detallado guardado como: {output_csv}")

if __name__ == "__main__":
    run_quality_test(dataset_path)