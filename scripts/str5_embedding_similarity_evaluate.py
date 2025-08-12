"""
Strategy 5: Enhanced Embedding-based Semantic Similarity Evaluation

This script evaluates LLM-generated knowledge graphs using sentence-transformers
with cosine similarity for more nuanced semantic matching.

Usage:
    python scripts/str5_embedding_similarity_evaluate.py --kg_name "mistral_basic_kg"
"""

import argparse
import os
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from kg_evaluator_module import KnowledgeGraphEvaluator

class EmbeddingSimilarityEvaluator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the embedding similarity evaluator."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        print(f"✓ Loaded sentence-transformer model: {model_name}")
    
    def triple_to_text(self, triple: Tuple[str, str, str]) -> str:
        """Convert triple to natural language text for embedding."""
        subject, predicate, obj = triple
        return f"{subject} {predicate} {obj}"
    
    def compute_similarity_matrix(self, llm_triples: List[Tuple[str, str, str]], 
                                ref_triples: List[Tuple[str, str, str]]) -> np.ndarray:
        """Compute similarity matrix between LLM and reference triples."""
        # Convert triples to text
        llm_texts = [self.triple_to_text(t) for t in llm_triples]
        ref_texts = [self.triple_to_text(t) for t in ref_triples]
        
        # Compute embeddings
        print("Computing embeddings for LLM triples...")
        llm_embeddings = self.model.encode(llm_texts, show_progress_bar=True, convert_to_numpy=True)
        
        print("Computing embeddings for reference triples...")
        ref_embeddings = self.model.encode(ref_texts, show_progress_bar=True, convert_to_numpy=True)
        
        # Compute cosine similarity matrix
        print("Computing similarity matrix...")
        # Convert to torch tensors if needed
        if not isinstance(llm_embeddings, torch.Tensor):
            llm_embeddings = torch.tensor(llm_embeddings)
        if not isinstance(ref_embeddings, torch.Tensor):
            ref_embeddings = torch.tensor(ref_embeddings)
        
        similarity_matrix = util.pytorch_cos_sim(llm_embeddings, ref_embeddings).numpy()
        
        return similarity_matrix
    
    def evaluate_triple_similarity(self, triple: Tuple[str, str, str], 
                                 ref_triples: List[Tuple[str, str, str]], 
                                 similarity_matrix: np.ndarray, 
                                 triple_idx: int) -> Dict[str, Any]:
        """Evaluate similarity for a single triple."""
        # Get similarity scores for this triple
        similarities = similarity_matrix[triple_idx]
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        best_ref_triple = ref_triples[best_idx]
        
        # Find all matches above threshold
        threshold = 0.7  # 70% similarity threshold
        matches = [(i, sim) for i, sim in enumerate(similarities) if sim >= threshold]
        
        # Get top 5 matches for analysis
        top_indices = np.argsort(similarities)[-5:][::-1]
        top_matches = [(ref_triples[i], similarities[i]) for i in top_indices]
        
        return {
            "triple": triple,
            "best_match": best_ref_triple,
            "best_similarity": float(best_similarity),
            "similarity_percentage": float(best_similarity * 100),
            "matches_above_threshold": len(matches),
            "top_matches": top_matches,
            "is_high_similarity": best_similarity >= threshold
        }
    
    def compute_aggregate_metrics(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate metrics from individual evaluations."""
        if not evaluations:
            return {}
        
        similarities = [e['best_similarity'] for e in evaluations]
        percentages = [e['similarity_percentage'] for e in evaluations]
        
        # Basic statistics
        avg_similarity = float(np.mean(similarities))
        avg_percentage = float(np.mean(percentages))
        std_similarity = float(np.std(similarities))
        
        # Threshold-based metrics
        high_similarity_count = sum(1 for e in evaluations if e['is_high_similarity'])
        high_similarity_rate = float(high_similarity_count / len(evaluations))
        
        # Similarity distribution
        similarity_ranges = {
            "very_high": sum(1 for s in similarities if s >= 0.9),  # 90%+
            "high": sum(1 for s in similarities if 0.7 <= s < 0.9),  # 70-90%
            "medium": sum(1 for s in similarities if 0.5 <= s < 0.7),  # 50-70%
            "low": sum(1 for s in similarities if 0.3 <= s < 0.5),  # 30-50%
            "very_low": sum(1 for s in similarities if s < 0.3)   # <30%
        }
        
        return {
            "total_triples": len(evaluations),
            "avg_similarity": avg_similarity,
            "avg_similarity_percentage": avg_percentage,
            "std_similarity": std_similarity,
            "high_similarity_rate": high_similarity_rate,
            "high_similarity_count": high_similarity_count,
            **{f"similarity_{k}": v for k, v in similarity_ranges.items()},
            "embedding_model": self.model_name
        }

def main():
    parser = argparse.ArgumentParser(description="Strategy 5: Enhanced embedding-based semantic similarity evaluation")
    parser.add_argument('--kg_name', required=True, help='Name of the LLM-generated KG in Neo4j')
    parser.add_argument('--sample_size', type=int, default=None, help='Number of triples to evaluate (default: all triples)')
    parser.add_argument('--embedding_model', default='all-MiniLM-L6-v2', 
                       help='Sentence-transformer model to use (default: all-MiniLM-L6-v2)')
    args = parser.parse_args()

    print(f"\n[Strategy 5] Evaluating KG '{args.kg_name}' using embedding-based semantic similarity...")
    print(f"Using embedding model: {args.embedding_model}")
    
    # Initialize evaluator
    evaluator = KnowledgeGraphEvaluator(args.kg_name, dataset_dir='data/ml1m-fixed', dataset_type='1m')
    embedding_evaluator = EmbeddingSimilarityEvaluator(args.embedding_model)
    
    try:
        # Load triples
        print("Loading triples from Neo4j...")
        llm_triples = list(evaluator.get_triples(args.kg_name))
        ref_triples = list(evaluator.get_triples('Golden KG (1m)'))
        
        print(f"LLM triples: {len(llm_triples)}")
        print(f"Reference triples: {len(ref_triples)}")
        
        # Sample triples for evaluation (only if sample_size is specified)
        if args.sample_size and len(llm_triples) > args.sample_size:
            import random
            random.seed(42)  # For reproducibility
            llm_triples = random.sample(llm_triples, args.sample_size)
            print(f"Sampled {len(llm_triples)} triples for evaluation")
        else:
            print(f"Evaluating all {len(llm_triples)} triples")
        
        # Compute similarity matrix
        similarity_matrix = embedding_evaluator.compute_similarity_matrix(llm_triples, ref_triples)
        
        # Evaluate each triple
        print("Evaluating triple similarities...")
        evaluations = []
        for i, triple in enumerate(tqdm(llm_triples, desc="Evaluating triples")):
            evaluation = embedding_evaluator.evaluate_triple_similarity(
                triple, ref_triples, similarity_matrix, i
            )
            evaluations.append(evaluation)
        
        # Compute aggregate metrics
        metrics = embedding_evaluator.compute_aggregate_metrics(evaluations)
        
        # Save results
        model_name = evaluator.model_name
        prompt_type = evaluator.prompt_type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed evaluations
        detailed_path = Path('KG_evaluation') / f"evaluate_{model_name}_{prompt_type}_str5_detailed_{timestamp}.json"
        os.makedirs('KG_evaluation', exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        detailed_data = []
        for eval in evaluations:
            eval_copy = eval.copy()
            # Convert numpy types to Python native types
            eval_copy['best_similarity'] = float(eval_copy['best_similarity'])
            eval_copy['similarity_percentage'] = float(eval_copy['similarity_percentage'])
            eval_copy['matches_above_threshold'] = int(eval_copy['matches_above_threshold'])
            eval_copy['is_high_similarity'] = bool(eval_copy['is_high_similarity'])
            eval_copy['top_matches'] = [(str(t), float(s)) for t, s in eval_copy['top_matches']]
            detailed_data.append(eval_copy)
        
        import json
        with open(detailed_path, 'w') as f:
            json.dump(detailed_data, f, indent=2)
        
        # Save summary metrics
        summary_path = Path('KG_evaluation') / f"evaluate_{model_name}_{prompt_type}_str5.csv"
        with open(summary_path, 'w') as f:
            f.write('metric,value\n')
            for k, v in metrics.items():
                f.write(f'{k},{v}\n')
        
        print(f"\nResults saved to:")
        print(f"  Summary: {summary_path}")
        print(f"  Detailed: {detailed_path}")
        
        print(f"\nSummary Metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        
        # Show sample high-similarity matches
        high_sim_triples = [e for e in evaluations if e['is_high_similarity']]
        if high_sim_triples:
            print(f"\nSample High-Similarity Matches (≥70%):")
            for i, eval in enumerate(high_sim_triples[:5]):
                print(f"  {i+1}. LLM: {eval['triple']}")
                print(f"     Ref:  {eval['best_match']}")
                print(f"     Sim:  {eval['similarity_percentage']:.1f}%")
                print()
            
    finally:
        evaluator.close()

if __name__ == "__main__":
    main() 