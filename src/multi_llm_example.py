"""
Flexible script for extracting a knowledge graph from any MovieLens-like dataset using multiple LLMs and prompt types.

Usage:
    python src/multi_llm_example.py --dataset data/ml-latest-small --max_samples 0 --models mistral llama2 --prompts basic complex

- --dataset: Path to the dataset directory (containing movies.csv and tags.csv)
- --max_samples: Number of samples to process (0 = all)
- --models: List of LLM model names to use (default: all)
- --prompts: List of prompt types to use (default: all)
"""

import argparse
from kg_extractor import KnowledgeGraphExtractor
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import sys
import csv
import os

def load_movielens_small(dataset_dir, max_samples=0):
    """Load and prepare MovieLens dataset from a given directory."""
    movies_path = Path(dataset_dir) / "movies.csv"
    tags_path = Path(dataset_dir) / "tags.csv"
    if not movies_path.exists() or not tags_path.exists():
        print(f"Error: movies.csv or tags.csv not found in {dataset_dir}")
        sys.exit(1)
    
    movies_df = pd.read_csv(movies_path)
    tags_df = pd.read_csv(tags_path)
    
    movie_texts = []
    for _, movie in movies_df.iterrows():
        # Filter out NaN values and convert tags to strings
        movie_tags = [str(tag) for tag in tags_df[tags_df['movieId'] == movie['movieId']]['tag'].tolist() if pd.notnull(tag)]
        text = f"{movie['title']} ({movie['title'].split('(')[-1].strip(')')}) is a {movie['genres'].replace('|', ', ')} film."
        if movie_tags:
            text += f" Tags: {', '.join(movie_tags)}."
        movie_texts.append(text)
    
    if max_samples and max_samples > 0:
        return movie_texts[:max_samples]
    return movie_texts

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
        # Extract year from title if present
        year = None
        if '(' in movie['title']:
            year = movie['title'].split('(')[-1].strip(')')
        
        # Build text description
        text = f"{movie['title']}"
        if year:
            text += f" ({year})"
        text += f" is a {movie['genres'].replace('|', ', ')} film."
        movie_texts.append(text)
    
    if max_samples and max_samples > 0:
        return movie_texts[:max_samples]
    return movie_texts

def main():
    parser = argparse.ArgumentParser(description="KG extraction from MovieLens-like datasets with LLMs.")
    parser.add_argument('--dataset', type=str, default='data/ml-1m', help='Path to dataset directory (default: data/ml-1m)')
    parser.add_argument('--dataset_type', type=str, default='1m', choices=['1m'], help='Dataset type: 1m (default: 1m)')
    parser.add_argument('--max_samples', type=int, default=0, help='Number of samples to process (0 = all)')
    parser.add_argument('--start_index', type=int, default=0, help='Index to start processing from (default: 0)')
    parser.add_argument('--models', nargs='+', default=['mistral', 'llama2', 'qwen', 'gemma', 'deepseek'], help='List of LLM model names to use (default: mistral, llama2, qwen, gemma, deepseek)')
    parser.add_argument('--prompts', nargs='+', default=['basic', 'complex', 'conversation', 'structured', 'cot', 'schema_guided', 'self_consistency'], help='List of prompt types to use (default: all). Options: basic, complex, conversation, structured, cot, schema_guided, self_consistency')
    args = parser.parse_args()

    # Hardcoded Together API key for all Together models
    together_api_key = "fe816936b359e9d975a42f152ceaf60328cb95f05b01a77adab28bcdd650945c"

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
                extractor = KnowledgeGraphExtractor(
                    model_name=model_name,
                    prompt_type=prompt_type,
                    together_api_key=together_api_key, # Only needed for Together models
                    dataset_type='1m' # Pass dataset_type for KG naming
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
    log_path = 'llm_kg_log.csv'
    if os.path.isfile(log_path):
        print('\nExperiment Summary:')
        summary = {}
        with open(log_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['model_name'], row['prompt_type'])
                if key not in summary:
                    summary[key] = {'input_tokens': 0, 'output_tokens': 0, 'latency': 0, 'num_triples': 0, 'count': 0}
                summary[key]['input_tokens'] += int(row['input_tokens'])
                summary[key]['output_tokens'] += int(row['output_tokens'])
                summary[key]['latency'] += float(row['latency'])
                summary[key]['num_triples'] += int(row['num_triples'])
                summary[key]['count'] += 1
        for (model, prompt_type), stats in summary.items():
            n = stats['count']
            print(f"Model: {model}, Prompt: {prompt_type}")
            print(f"  Samples: {n}")
            print(f"  Total input tokens: {stats['input_tokens']}")
            print(f"  Total output tokens: {stats['output_tokens']}")
            print(f"  Total triples: {stats['num_triples']}")
            print(f"  Avg latency: {stats['latency']/n:.2f} s")
            print(f"  Avg input tokens: {stats['input_tokens']/n:.1f}")
            print(f"  Avg output tokens: {stats['output_tokens']/n:.1f}")
            print(f"  Avg triples: {stats['num_triples']/n:.1f}\n")
    print("\nAll done! Check your Neo4j database for the extracted knowledge graph.")

if __name__ == "__main__":
    main() 