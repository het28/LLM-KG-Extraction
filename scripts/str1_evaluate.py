import argparse
from datetime import datetime
from pathlib import Path
import os
from tqdm import tqdm
from kg_evaluator_module import KnowledgeGraphEvaluator

def load_triples_from_tsv(tsv_path):
    triples = set()
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                triples.add(tuple(parts))
    return triples

def main():
    parser = argparse.ArgumentParser(description="Strategy 1: Evaluate LLM KG against DBpedia ground truth (exact match)")
    parser.add_argument('--kg_name', required=True, help='Name of the LLM-generated KG in Neo4j')
    args = parser.parse_args()

    print(f"\n[Strategy 1] Evaluating KG '{args.kg_name}' against DBpedia ground truth (exact match, from TSV)...")
    evaluator = KnowledgeGraphEvaluator(args.kg_name, dataset_dir='data/ml1m-fixed', dataset_type='1m')
    try:
        print("Fetching triples from Neo4j and DBpedia ground truth TSV file...")
        llm_triples = evaluator.get_triples(args.kg_name)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tsv_path = os.path.join(script_dir, '..', 'data', 'ml1m_fixed', 'kg_triples_names.tsv')
        ref_triples = load_triples_from_tsv(tsv_path)
        print(f"  LLM KG triples: {len(llm_triples)}")
        print(f"  DBpedia ground truth triples (from TSV): {len(ref_triples)}")

        print("\nComparing triples (exact match)...")
        true_positives = set()
        for triple in tqdm(llm_triples, desc="Checking true positives"):
            if triple in ref_triples:
                true_positives.add(triple)
        false_positives = llm_triples - ref_triples
        false_negatives = ref_triples - llm_triples
        precision = len(true_positives) / len(llm_triples) if llm_triples else 0.0
        recall = len(true_positives) / len(ref_triples) if ref_triples else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": len(true_positives),
            "false_positives": len(false_positives),
            "false_negatives": len(false_negatives),
            "llm_triples": len(llm_triples),
            "ref_triples": len(ref_triples)
        }
        model_name = evaluator.model_name
        prompt_type = evaluator.prompt_type
        out_path = Path('KG_evaluation') / f"evaluate_{model_name}_{prompt_type}_str1.csv"
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