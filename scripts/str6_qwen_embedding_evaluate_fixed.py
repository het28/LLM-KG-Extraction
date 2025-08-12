"""
Strategy 6: Qwen Embedding-based Evaluation (Fixed)

This script evaluates LLM-generated knowledge graphs using official Qwen3-Embedding models
via Hugging Face transformers for comparison with sentence-transformer approaches.

Usage:
    python scripts/str6_qwen_embedding_evaluate_fixed.py --kg_name "mistral_basic_kg"
"""

import argparse
import os
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from kg_evaluator_module import KnowledgeGraphEvaluator

class QwenEmbeddingEvaluator:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B"):
        """Initialize the Qwen embedding evaluator using Hugging Face transformers."""
        self.model_name = model_name
        
        print(f"Loading Qwen3-Embedding model: {model_name}")
        
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ Qwen3-Embedding model {model_name} loaded successfully")
            
        except Exception as e:
            print(f"Error loading Qwen3-Embedding model: {e}")
            print("Falling back to sentence-transformers...")
            self._fallback_to_sentence_transformers()
    
    def _fallback_to_sentence_transformers(self):
        """Fallback to sentence-transformers if Qwen model fails to load."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.model_name = "all-MiniLM-L6-v2 (fallback)"
            print(f"✓ Using fallback model: {self.model_name}")
        except Exception as e:
            print(f"Error loading fallback model: {e}")
            raise
    
    def triple_to_text(self, triple: Tuple[str, str, str]) -> str:
        """Convert triple to natural language text for embedding."""
        subject, predicate, obj = triple
        return f"{subject} {predicate} {obj}"
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts using Qwen3-Embedding model."""
        try:
            if hasattr(self.model, 'encode'):  # SentenceTransformer fallback
                embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
                return embeddings
            
            # Qwen3-Embedding model via transformers
            embeddings = []
            batch_size = 32  # Process in batches to avoid memory issues
            
            for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use mean pooling of last hidden states
                    attention_mask = inputs['attention_mask']
                    embeddings_batch = self._mean_pooling(outputs.last_hidden_state, attention_mask)
                    embeddings.append(embeddings_batch.cpu().numpy().astype(np.float32))
            
            return np.vstack(embeddings)
            
        except Exception as e:
            print(f"Error computing embeddings: {e}")
            # Return zero vectors as fallback
            embedding_dim = 768  # Default embedding dimension
            return np.zeros((len(texts), embedding_dim))
    
    def _mean_pooling(self, token_embeddings, attention_mask):
        """Compute mean pooling of token embeddings."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def compute_similarity_matrix(self, llm_triples: List[Tuple[str, str, str]], 
                                ref_triples: List[Tuple[str, str, str]]) -> np.ndarray:
        """Compute similarity matrix between LLM and reference triples."""
        # Convert triples to text
        llm_texts = [self.triple_to_text(t) for t in llm_triples]
        ref_texts = [self.triple_to_text(t) for t in ref_triples]
        
        # Compute embeddings
        print("Computing Qwen3-Embedding for LLM triples...")
        llm_embeddings = self.get_embeddings(llm_texts)
        
        print("Computing Qwen3-Embedding for reference triples...")
        ref_embeddings = self.get_embeddings(ref_texts)
        
        # Compute cosine similarity matrix using manual computation
        print("Computing similarity matrix using cosine similarity...")
        similarity_matrix = np.zeros((len(llm_triples), len(ref_triples)))
        
        for i in tqdm(range(len(llm_triples)), desc="Computing similarities"):
            for j in range(len(ref_triples)):
                similarity_matrix[i, j] = self.cosine_similarity(
                    llm_embeddings[i], ref_embeddings[j]
                )
        
        return similarity_matrix
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def evaluate_triple_similarity(self, triple: Tuple[str, str, str], 
                                 ref_triples: List[Tuple[str, str, str]], 
                                 similarity_matrix: np.ndarray, 
                                 triple_idx: int) -> Dict[str, Any]:
        """Evaluate similarity for a single triple."""
        similarities = similarity_matrix[triple_idx]
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        best_ref_triple = ref_triples[best_idx]
        
        # Find all matches above threshold
        threshold = 0.7
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
        avg_similarity = np.mean(similarities)
        avg_percentage = np.mean(percentages)
        std_similarity = np.std(similarities)
        
        # Threshold-based metrics
        high_similarity_count = sum(1 for e in evaluations if e['is_high_similarity'])
        high_similarity_rate = high_similarity_count / len(evaluations)
        
        # Similarity distribution
        similarity_ranges = {
            "very_high": sum(1 for s in similarities if s >= 0.9),
            "high": sum(1 for s in similarities if 0.7 <= s < 0.9),
            "medium": sum(1 for s in similarities if 0.5 <= s < 0.7),
            "low": sum(1 for s in similarities if 0.3 <= s < 0.5),
            "very_low": sum(1 for s in similarities if s < 0.3)
        }
        
        return {
            "total_triples": float(len(evaluations)),
            "avg_similarity": float(avg_similarity),
            "avg_similarity_percentage": float(avg_percentage),
            "std_similarity": float(std_similarity),
            "high_similarity_rate": float(high_similarity_rate),
            "high_similarity_count": float(high_similarity_count),
            **{f"similarity_{k}": float(v) for k, v in similarity_ranges.items()},
            "embedding_model": str(self.model_name)
        }

def main():
    parser = argparse.ArgumentParser(description="Strategy 6: Qwen3-Embedding-based evaluation")
    parser.add_argument('--kg_name', required=True, help='Name of the LLM-generated KG in Neo4j')
    parser.add_argument('--sample_size', type=int, default=None, help='Number of triples to evaluate (default: all triples)')
    parser.add_argument('--embedding_model', default='Qwen/Qwen3-Embedding-0.6B', help='Qwen3-Embedding model to use')
    args = parser.parse_args()

    print(f"\n[Strategy 6] Evaluating KG '{args.kg_name}' using Qwen3-Embedding...")
    print(f"Using Qwen3-Embedding model: {args.embedding_model}")
    
    # Initialize evaluator
    evaluator = KnowledgeGraphEvaluator(args.kg_name, dataset_dir='data/ml1m-fixed', dataset_type='1m')
    qwen_evaluator = QwenEmbeddingEvaluator(args.embedding_model)
    
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
            random.seed(42)
            llm_triples = random.sample(llm_triples, args.sample_size)
            print(f"Sampled {len(llm_triples)} triples for evaluation")
        else:
            print(f"Evaluating all {len(llm_triples)} triples")
        
        # Compute similarity matrix
        similarity_matrix = qwen_evaluator.compute_similarity_matrix(llm_triples, ref_triples)
        
        # Evaluate each triple
        print("Evaluating triple similarities...")
        evaluations = []
        for i, triple in enumerate(tqdm(llm_triples, desc="Evaluating triples")):
            evaluation = qwen_evaluator.evaluate_triple_similarity(
                triple, ref_triples, similarity_matrix, i
            )
            evaluations.append(evaluation)
        
        # Compute aggregate metrics
        metrics = qwen_evaluator.compute_aggregate_metrics(evaluations)
        
        # Save results
        model_name = evaluator.model_name
        prompt_type = evaluator.prompt_type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed evaluations
        detailed_path = Path('KG_evaluation') / f"evaluate_{model_name}_{prompt_type}_str6_qwen_detailed_{timestamp}.json"
        os.makedirs('KG_evaluation', exist_ok=True)
        
        # Convert numpy types to Python native types for JSON serialization
        detailed_data = []
        for eval in evaluations:
            eval_copy = eval.copy()
            eval_copy['best_similarity'] = float(eval_copy['best_similarity'])
            eval_copy['similarity_percentage'] = float(eval_copy['similarity_percentage'])
            eval_copy['matches_above_threshold'] = int(eval_copy['matches_above_threshold'])
            eval_copy['is_high_similarity'] = bool(eval_copy['is_high_similarity'])
            eval_copy['top_matches'] = [(str(t), float(s)) for t, s in eval_copy['top_matches']]
            detailed_data.append(eval_copy)
        
        with open(detailed_path, 'w') as f:
            json.dump(detailed_data, f, indent=2)
        
        # Save summary metrics
        summary_path = Path('KG_evaluation') / f"evaluate_{model_name}_{prompt_type}_str6_qwen.csv"
        with open(summary_path, 'w') as f:
            f.write('metric,value\n')
            for k, v in metrics.items():
                f.write(f'{k},{v}\n')
        
        print(f"\nResults saved to:")
        print(f"  Summary: {summary_path}")
        print(f"  Detailed: {detailed_path}")
        
        print(f"\nSummary Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
            
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        raise
    finally:
        evaluator.close()

if __name__ == "__main__":
    main() 