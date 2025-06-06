"""
Advanced script for extracting a knowledge graph using multiple LLMs with self-consistency voting.

Usage:
    python src/advanced_multi_llm_example.py --dataset data/ml-1m --max_samples 0 --models mistral llama2 --prompts basic complex

- --dataset: Path to the dataset directory (containing movies.dat)
- --max_samples: Number of samples to process (0 = all)
- --models: List of LLM model names to use (default: mistral, llama2, qwen, gemma, deepseek)
- --prompts: List of prompt types to use (default: all)
- --iterations: Number of extraction iterations for self-consistency (default: 3)
- --voting_threshold: Minimum agreement threshold for triple acceptance (default: 0.7)
- --dataset_type: Dataset type: 1m (default: 1m)
"""

import argparse
from advanced_kg_extractor import AdvancedKnowledgeGraphExtractor
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import sys
import csv
import os

def load_movielens_1m(dataset_dir, max_samples=0):
    """Load and prepare MovieLens 1M dataset from a given directory."""
    movies_path = Path(dataset_dir) / "movies.dat"
    if not movies_path.exists():
        print(f"Error: movies.dat not found in {dataset_dir}")
        sys.exit(1)
    # Read movies.dat with proper encoding and separator
    movies_df = pd.read_csv(movies_path, sep='::', engine='python', 
                           names=['movieId', 'title', 'genres'], 
                           encoding='latin-1')
    movie_texts = []
    for _, movie in movies_df.iterrows():
        year = None
        if '(' in movie['title']:
            year = movie['title'].split('(')[-1].strip(')')
        text = f"{movie['title']}"
        if year:
            text += f" ({year})"
        text += f" is a {movie['genres'].replace('|', ', ')} film."
        movie_texts.append(text)
    if max_samples and max_samples > 0:
        return movie_texts[:max_samples]
    return movie_texts

def main():
    parser = argparse.ArgumentParser(description="Advanced KG extraction with self-consistency voting.")
    parser.add_argument('--dataset', type=str, default='data/ml-1m', 
                       help='Path to dataset directory (default: data/ml-1m)')
    parser.add_argument('--max_samples', type=int, default=0, 
                       help='Number of samples to process (0 = all)')
    parser.add_argument('--models', nargs='+', default=['mistral', 'llama2', 'qwen', 'gemma', 'deepseek', 'granite'], 
                       help='List of LLM model names to use (default: mistral, llama2, qwen, gemma, deepseek, granite)')
    parser.add_argument('--prompts', nargs='+', 
                       default=['basic', 'complex', 'conversation', 'structured', 'cot', 'schema_guided', 'self_consistency'], 
                       help='List of prompt types to use (default: all)')
    parser.add_argument('--iterations', type=int, default=3,
                       help='Number of extraction iterations for self-consistency (default: 3)')
    parser.add_argument('--voting_threshold', type=float, default=0.7,
                       help='Minimum agreement threshold for triple acceptance (default: 0.7)')
    parser.add_argument('--dataset_type', type=str, default='1m', choices=['1m'], help='Dataset type: 1m (default: 1m)')
    parser.add_argument('--start_index', type=int, default=0, help='Index to start processing from (default: 0)')
    args = parser.parse_args()

    # Validate prompt types
    valid_prompts = {'basic', 'complex', 'conversation', 'structured', 'cot', 'schema_guided', 'self_consistency'}
    for p in args.prompts:
        if p not in valid_prompts:
            print(f"Error: Invalid prompt type '{p}'. Valid options: {valid_prompts}")
            sys.exit(1)

    # Load MovieLens 1M data
    print(f"Loading data from {args.dataset} ...")
    movie_texts = load_movielens_1m(args.dataset, args.max_samples)
    print(f"Loaded {len(movie_texts)} movie descriptions.")
    
    # Slice movie_texts if start_index is specified
    if args.start_index > 0:
        movie_texts = movie_texts[args.start_index:]
        print(f"Resuming from description #{args.start_index + 1} (0-based index {args.start_index})...")
    
    extractors = []
    try:
        for model_name in args.models:
            for prompt_type in args.prompts:
                print(f"\nInitializing {model_name} with {prompt_type} prompt...")
                extractor = AdvancedKnowledgeGraphExtractor(
                    model_name=model_name,
                    prompt_type=prompt_type,
                    num_iterations=args.iterations,
                    voting_threshold=args.voting_threshold,
                    dataset_type='1m'
                )
                extractors.append(extractor)
                
                print(f"Processing {len(movie_texts)} texts with {model_name} ({prompt_type} prompt)...")
                for text in tqdm(movie_texts):
                    triples = extractor.extract_triples(text)
                    for subject, predicate, obj in triples:
                        extractor.add_triple(subject, predicate, obj)
                print(f"Finished {model_name} ({prompt_type})!")
    finally:
        for extractor in extractors:
            extractor.close()
            
    # Print experiment summary
    print('\nExperiment Summary:')
    for model_name in args.models:
        for prompt_type in args.prompts:
            log_path = f"advance_{model_name}_{prompt_type}.csv"
            if os.path.isfile(log_path):
                summary = {'input_tokens': 0, 'output_tokens': 0, 'latency': 0, 'num_triples': 0, 'count': 0}
                with open(log_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        summary['input_tokens'] += int(row['input_tokens'])
                        summary['output_tokens'] += int(row['output_tokens'])
                        summary['latency'] += float(row['latency'])
                        summary['num_triples'] += int(row['num_triples'])
                        summary['count'] += 1
                
                n = summary['count']
                print(f"\nModel: {model_name}, Prompt: {prompt_type}")
                print(f"  Samples: {n}")
                print(f"  Total input tokens: {summary['input_tokens']}")
                print(f"  Total output tokens: {summary['output_tokens']}")
                print(f"  Total triples: {summary['num_triples']}")
                print(f"  Avg latency: {summary['latency']/n:.2f} s")
                print(f"  Avg input tokens: {summary['input_tokens']/n:.1f}")
                print(f"  Avg output tokens: {summary['output_tokens']/n:.1f}")
                print(f"  Avg triples: {summary['num_triples']/n:.1f}")
                print(f"  Iterations: {args.iterations}")
                print(f"  Voting threshold: {args.voting_threshold}")
    
    print("\nAll done! Check your Neo4j database for the extracted knowledge graph.")

if __name__ == "__main__":
    main() 