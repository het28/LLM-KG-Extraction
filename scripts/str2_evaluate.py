import argparse
from datetime import datetime
from pathlib import Path
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from kg_evaluator_module import KnowledgeGraphEvaluator
import numpy as np

def triple_to_str(triple):
    return f"{triple[0]} | {triple[1]} | {triple[2]}"

def main():
    parser = argparse.ArgumentParser(description="Strategy 2: Evaluate LLM KG against Golden KG (semantic/NER, soft matching)")
    parser.add_argument('--kg_name', required=True, help='Name of the LLM-generated KG in Neo4j')
    args = parser.parse_args()

    print(f"\n[Strategy 2] Evaluating KG '{args.kg_name}' against Golden KG (semantic/NER, soft matching)...")
    evaluator = KnowledgeGraphEvaluator(args.kg_name, dataset_dir='data/ml1m-fixed', dataset_type='1m')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    try:
        print("Fetching triples from Neo4j and Golden KG...")
        llm_triples = evaluator.get_triples(args.kg_name)
        ref_triples = evaluator.get_triples('Golden KG (1m)')
        print(f"  LLM KG triples: {len(llm_triples)}")
        print(f"  Golden KG triples: {len(ref_triples)}")

        print("\nComputing semantic similarity for soft matching...")
        llm_triples_list = list(llm_triples)
        ref_triples_list = list(ref_triples)
        llm_strs = [triple_to_str(t) for t in llm_triples_list]
        ref_strs = [triple_to_str(t) for t in ref_triples_list]
        llm_emb = model.encode(llm_strs, show_progress_bar=True, convert_to_numpy=True)
        ref_emb = model.encode(ref_strs, show_progress_bar=True, convert_to_numpy=True)
        sim_matrix = np.matmul(llm_emb, ref_emb.T)
        # Normalize
        llm_norm = np.linalg.norm(llm_emb, axis=1, keepdims=True)
        ref_norm = np.linalg.norm(ref_emb, axis=1, keepdims=True)
        sim_matrix = sim_matrix / (llm_norm + 1e-8)
        sim_matrix = sim_matrix / (ref_norm.T + 1e-8)
        # For each LLM triple, find best match in Golden KG
        threshold = 0.8
        soft_true_positives = 0
        total_similarity = 0.0
        matched_ref = set()
        for i, row in enumerate(tqdm(sim_matrix, desc="Soft matching LLM triples")):
            best_j = np.argmax(row)
            best_sim = row[best_j]
            if best_sim >= threshold:
                soft_true_positives += 1
                matched_ref.add(best_j)
                total_similarity += best_sim
        soft_false_positives = len(llm_triples) - soft_true_positives
        soft_false_negatives = len(ref_triples) - len(matched_ref)
        soft_precision = soft_true_positives / len(llm_triples) if llm_triples else 0.0
        soft_recall = soft_true_positives / len(ref_triples) if ref_triples else 0.0
        soft_f1 = 2 * soft_precision * soft_recall / (soft_precision + soft_recall) if (soft_precision + soft_recall) > 0 else 0.0
        avg_similarity = total_similarity / soft_true_positives if soft_true_positives > 0 else 0.0
        metrics = {
            "soft_precision": soft_precision,
            "soft_recall": soft_recall,
            "soft_f1": soft_f1,
            "avg_similarity": avg_similarity,
            "soft_true_positives": soft_true_positives,
            "soft_false_positives": soft_false_positives,
            "soft_false_negatives": soft_false_negatives,
            "llm_triples": len(llm_triples),
            "ref_triples": len(ref_triples)
        }
        model_name = evaluator.model_name
        prompt_type = evaluator.prompt_type
        out_path = Path('KG_evaluation') / f"evaluate_{model_name}_{prompt_type}_str2.csv"
        os.makedirs('KG_evaluation', exist_ok=True)
        with open(out_path, 'w') as f:
            f.write('metric,value\n')
            for k, v in metrics.items():
                f.write(f'{k},{v}\n')
        print(f"\nResults saved to {out_path}\n")
        print("Summary:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    finally:
        evaluator.close()

if __name__ == '__main__':
    main() 