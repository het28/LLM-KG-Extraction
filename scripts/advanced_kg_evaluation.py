"""
Advanced Knowledge Graph Evaluation Script

This script evaluates LLM-generated knowledge graphs against a golden KG using:
1. Traditional metrics (precision, recall, F1)
2. Graph-based metrics (subgraph matching, backbone comparison)
3. Path-based metrics (path overlap, path similarity)
4. Hallucination detection (triples not supported by source text)
5. Extraction efficiency metrics (from CSV logs)
6. Entity mapping and normalization
7. Soft matching with semantic similarity
8. Schema/ontology conformance checking

Usage:
    python scripts/advanced_kg_evaluation.py --kg_name "mistral_basic_kg" --dataset "data/ml-1m"

- --kg_name: Name of the LLM-generated KG to evaluate
- --dataset: Path to MovieLens dataset directory
"""

import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Advanced KG evaluation: choose strategy and KG name.")
    parser.add_argument('--strategy', required=True, choices=['str1', 'str2', 'str3'], help='Evaluation strategy to use')
    parser.add_argument('--kg_name', required=True, help='Name of the LLM-generated KG in Neo4j')
    args = parser.parse_args()

    script_map = {
        'str1': 'str1_evaluate.py',
        'str2': 'str2_evaluate.py',
        'str3': 'str3_evaluate.py',
    }
    script = script_map[args.strategy]
    print(f"\n[Dispatcher] Running {script} for KG '{args.kg_name}'...")
    try:
        result = subprocess.run([sys.executable, script, '--kg_name', args.kg_name], cwd='scripts', capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("[stderr]", result.stderr)
    except Exception as e:
        print(f"Error running {script}: {e}")

if __name__ == '__main__':
    main() 
