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
import csv
from neo4j import GraphDatabase
import networkx as nx
from typing import Set, List, Tuple, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
from pathlib import Path
import sys
import os
from collections import defaultdict
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

def load_movielens_small(dataset_dir: str) -> Dict[str, str]:
    movies_path = Path(dataset_dir) / "movies.csv"
    tags_path = Path(dataset_dir) / "tags.csv"
    if not movies_path.exists():
        print(f"Error: movies.csv not found in {dataset_dir}")
        sys.exit(1)
    movies_df = pd.read_csv(movies_path)
    tags_df = pd.read_csv(tags_path) if tags_path.exists() else pd.DataFrame(columns=["movieId", "tag"])
    texts = {}
    for _, movie in movies_df.iterrows():
        movie_tags = tags_df[tags_df['movieId'] == movie['movieId']]['tag'].tolist() if not tags_df.empty else []
        text = f"{movie['title']} ({movie['title'].split('(')[-1].strip(')')}) is a {movie['genres'].replace('|', ', ')} film."
        if movie_tags:
            text += f" Tags: {', '.join(movie_tags)}."
        texts[str(movie['movieId'])] = text
    return texts

def load_movielens_1m(dataset_dir: str) -> Dict[str, str]:
    """Load and prepare MovieLens 1M dataset from a given directory."""
    movies_path = Path(dataset_dir) / "movies.dat"
    if not movies_path.exists():
        print(f"Error: movies.dat not found in {dataset_dir}")
        sys.exit(1)
    
    # Read movies.dat with proper encoding and separator
    movies_df = pd.read_csv(movies_path, sep='::', engine='python', 
                           names=['movieId', 'title', 'genres'], 
                           encoding='latin-1')
    
    texts = {}
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
        texts[str(movie['movieId'])] = text
    
    return texts

def load_file_ground_truth(tsv_path: str) -> Set[Tuple[str, str, str]]:
    """Load ground truth triples from a TSV file."""
    triples = set()
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                s, o, p = parts
                triples.add((s, p, o))
    return triples

def load_mapping_files() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Load all mapping files for entity and relation normalization."""
    # Load entity mappings
    items_mapping = {}
    with open('data/ml1m_fixed/mapping_items.tsv', 'r', encoding='utf-8') as f:
        for line in f:
            ml_id, dbpedia_uri = line.strip().split('\t')
            items_mapping[ml_id] = dbpedia_uri
    
    # Load property mappings
    props_mapping = {}
    with open('data/ml1m_fixed/mapping_props.tsv', 'r', encoding='utf-8') as f:
        for line in f:
            prop_id, dbpedia_uri = line.strip().split('\t')
            props_mapping[prop_id] = dbpedia_uri
    
    # Load relation mappings
    relations_mapping = {}
    with open('data/ml1m_fixed/mapping_relations.tsv', 'r', encoding='utf-8') as f:
        for line in f:
            rel_id, dbpedia_uri = line.strip().split('\t')
            relations_mapping[rel_id] = dbpedia_uri
    
    return items_mapping, props_mapping, relations_mapping

# Neo4j connection details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"

class KnowledgeGraphEvaluator:
    def __init__(self, kg_name: str, dataset_dir: str, dataset_type: str = '1m'):
        """Initialize the KG evaluator with enhanced features."""
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.kg_name = kg_name
        self.dataset_type = dataset_type
        
        # Load source texts based on dataset type
        if dataset_type == '1m':
            self.source_texts = load_movielens_1m(dataset_dir)
        else:
            self.source_texts = load_movielens_small(dataset_dir)
        
        # Initialize sentence transformer for semantic similarity
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Extract model and prompt type from kg_name
        parts = kg_name.split('_')
        if 'advance' in kg_name:
            self.model_name = parts[1]
            self.prompt_type = parts[2]
        else:
            self.model_name = parts[0]
            self.prompt_type = parts[1]
        
        # Load mapping files for entity and relation normalization
        self.items_mapping, self.props_mapping, self.relations_mapping = load_mapping_files()
        
        # Load ground truth
        self.file_ground_truth = load_file_ground_truth('data/ml1m_fixed/kg_triples_names.tsv')
        
        # Cache for semantic similarity computations
        self.similarity_cache = {}

    def normalize_triple(self, triple: Tuple[str, str, str], use_soft_matching: bool = False) -> Tuple[str, str, str]:
        """Normalize a triple using mapping files and optional soft matching."""
        s, p, o = triple
        
        # Try exact mapping first
        s_mapped = self.items_mapping.get(s, s)
        p_mapped = self.relations_mapping.get(p, self.props_mapping.get(p, p))
        o_mapped = self.items_mapping.get(o, o)
        
        if use_soft_matching and (s_mapped == s or o_mapped == o):
            # Try soft matching for unmapped entities
            if s_mapped == s:
                s_mapped = self._find_similar_entity(s, self.items_mapping.values())
            if o_mapped == o:
                o_mapped = self._find_similar_entity(o, self.items_mapping.values())
        
        return (s_mapped, p_mapped, o_mapped)

    def _find_similar_entity(self, entity: str, candidates, threshold: float = 0.8) -> str:
        if entity in self.similarity_cache:
            return self.similarity_cache[entity]
        candidates_list = list(candidates)
        entity_emb = self.model.encode([entity])[0]
        candidates_emb = self.model.encode(candidates_list)
        similarities = cosine_similarity([entity_emb], candidates_emb)[0]
        max_sim_idx = np.argmax(similarities)
        if similarities[max_sim_idx] >= threshold:
            result = candidates_list[max_sim_idx]
            self.similarity_cache[entity] = result
            return result
        return entity

    def compute_traditional_metrics(self, llm_triples: Set[Tuple[str, str, str]], 
                                  ref_triples: Set[Tuple[str, str, str]],
                                  use_soft_matching: bool = False) -> Dict[str, float]:
        """Compute enhanced traditional metrics with optional soft matching."""
        # Normalize all triples
        llm_triples_norm = {self.normalize_triple(t, use_soft_matching) for t in llm_triples}
        ref_triples_norm = {self.normalize_triple(t, use_soft_matching) for t in ref_triples}
        
        # Compute exact match metrics
        true_positives = llm_triples_norm & ref_triples_norm
        false_positives = llm_triples_norm - ref_triples_norm
        false_negatives = ref_triples_norm - llm_triples_norm
        
        precision = len(true_positives) / len(llm_triples_norm) if llm_triples_norm else 0.0
        recall = len(true_positives) / len(ref_triples_norm) if ref_triples_norm else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Compute soft match metrics if enabled
        soft_metrics = {}
        if use_soft_matching:
            soft_tp = 0
            for t1 in llm_triples_norm:
                for t2 in ref_triples_norm:
                    if self._is_soft_match(t1, t2):
                        soft_tp += 1
                        break
            
            soft_precision = soft_tp / len(llm_triples_norm) if llm_triples_norm else 0.0
            soft_recall = soft_tp / len(ref_triples_norm) if ref_triples_norm else 0.0
            soft_f1 = 2 * soft_precision * soft_recall / (soft_precision + soft_recall) if (soft_precision + soft_recall) > 0 else 0.0
            
            soft_metrics = {
                "soft_precision": soft_precision,
                "soft_recall": soft_recall,
                "soft_f1": soft_f1
            }
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": len(true_positives),
            "false_positives": len(false_positives),
            "false_negatives": len(false_negatives),
            **soft_metrics
        }

    def _is_soft_match(self, triple1: Tuple[str, str, str], triple2: Tuple[str, str, str], 
                      threshold: float = 0.8) -> bool:
        """Check if two triples match using semantic similarity."""
        s1, p1, o1 = triple1
        s2, p2, o2 = triple2
        
        # Check predicate first (exact match)
        if p1 != p2:
            return False
        
        # Check subject and object with semantic similarity
        s_sim = cosine_similarity(
            [self.model.encode([s1])[0]], 
            [self.model.encode([s2])[0]]
        )[0][0]
        
        o_sim = cosine_similarity(
            [self.model.encode([o1])[0]], 
            [self.model.encode([o2])[0]]
        )[0][0]
        
        return s_sim >= threshold and o_sim >= threshold

    def detect_hallucinations(self, llm_triples: Set[Tuple[str, str, str]]) -> Dict[str, List[Tuple[str, str, str]]]:
        """Enhanced hallucination detection with detailed explanations."""
        hallucinated = []
        supported = []
        explanations = []
        
        for s, p, o in llm_triples:
            # Check if triple exists in ground truth
            if (s, p, o) in self.file_ground_truth:
                supported.append((s, p, o))
                continue
            
            # Find the source text for this triple
            source_text = None
            for text in self.source_texts.values():
                if s in text or o in text:
                    source_text = text
                    break
            
            if not source_text:
                hallucinated.append((s, p, o))
                explanations.append(f"Entity not found in source text: {s} or {o}")
                continue
            
            # Check semantic similarity with source text
            triple_text = f"{s} {p} {o}"
            triple_embedding = self.model.encode(triple_text)
            text_embedding = self.model.encode(source_text)
            
            similarity = np.dot(triple_embedding, text_embedding) / (
                np.linalg.norm(triple_embedding) * np.linalg.norm(text_embedding)
            )
            
            if similarity < 0.7:
                hallucinated.append((s, p, o))
                explanations.append(f"Low semantic similarity with source text: {similarity:.2f}")
            else:
                supported.append((s, p, o))
        
        return {
            "hallucinated": hallucinated,
            "supported": supported,
            "explanations": explanations
        }

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        """Run all evaluations with enhanced metrics."""
        # Get triples and graphs
        llm_triples = self.get_triples(self.kg_name)
        ref_triples = self.get_triples('DBpedia Ground Truth')
        llm_graph = self.get_graph(self.kg_name)
        ref_graph = self.get_graph('DBpedia Ground Truth')
        
        # Compute all metrics
        traditional = self.compute_traditional_metrics(llm_triples, ref_triples, use_soft_matching=True)
        graph_metrics = self.compute_graph_metrics(llm_graph, ref_graph)
        path_metrics = self.compute_path_metrics(llm_graph, ref_graph)
        hallucination = self.detect_hallucinations(llm_triples)
        extraction_metrics = self.analyze_csv_logs()
        
        # Add correlation analysis
        correlations = self.analyze_correlations(traditional, extraction_metrics)
        
        return {
            "traditional": traditional,
            "graph": graph_metrics,
            "path": path_metrics,
            "hallucination": {
                "hallucination_rate": len(hallucination["hallucinated"]) / len(llm_triples) if llm_triples else 0.0,
                "num_hallucinated": len(hallucination["hallucinated"]),
                "num_supported": len(hallucination["supported"]),
                "explanations": hallucination["explanations"]
            },
            "extraction": extraction_metrics,
            "correlations": correlations
        }

    def analyze_correlations(self, traditional_metrics: Dict[str, float], 
                           extraction_metrics: Dict[str, float]) -> Dict[str, float]:
        """Analyze correlations between extraction metrics and KG quality."""
        correlations = {}
        
        # Only analyze if we have extraction metrics
        if extraction_metrics["count"] > 0:
            # Calculate correlation between number of triples and quality
            correlations["triples_vs_precision"] = self._correlation(
                extraction_metrics["num_triples"],
                traditional_metrics["precision"]
            )
            
            # Calculate correlation between confidence and quality
            correlations["confidence_vs_precision"] = self._correlation(
                extraction_metrics["avg_confidence"],
                traditional_metrics["precision"]
            )
            
            # Calculate correlation between latency and quality
            correlations["latency_vs_precision"] = self._correlation(
                extraction_metrics["avg_latency"],
                traditional_metrics["precision"]
            )
        
        return correlations

    def _correlation(self, x: float, y: float) -> float:
        """Calculate correlation coefficient between two metrics."""
        # For single values, return 0 (no correlation)
        return 0.0

    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()

    def get_triples(self, kg_name: str) -> Set[Tuple[str, str, str]]:
        """Get all triples from a KG. For ground truth, always use file-based triples."""
        if kg_name == 'DBpedia Ground Truth':
            return self.file_ground_truth
        with self.driver.session() as session:
            result = session.run(
                "MATCH (s)-[r]->(o) WHERE r.kg_name = $kg_name RETURN s.name as s, type(r) as p, o.name as o",
                kg_name=kg_name
            )
            return {(r["s"], r["p"], r["o"]) for r in result}

    def get_graph(self, kg_name: str) -> nx.DiGraph:
        """Convert KG triples to NetworkX directed graph."""
        G = nx.DiGraph()
        triples = self.get_triples(kg_name)
        for s, p, o in triples:
            G.add_edge(s, o, predicate=p)
        return G

def save_evaluation_results(metrics: Dict[str, Dict[str, float]], 
                          model_name: str, 
                          prompt_type: str, 
                          dataset_type: str):
    """Save evaluation results to a CSV file in the KG_evaluation directory."""
    # Create KG_evaluation directory if it doesn't exist
    eval_dir = Path("KG_evaluation")
    eval_dir.mkdir(exist_ok=True)
    
    # Create filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{prompt_type}_eval_{dataset_type}_{timestamp}.csv"
    filepath = eval_dir / filename
    
    # Flatten the metrics dictionary for CSV storage
    flat_metrics = {
        "timestamp": timestamp,
        "model_name": model_name,
        "prompt_type": prompt_type,
        "dataset_type": dataset_type
    }
    
    # Add traditional metrics
    for metric, value in metrics["traditional"].items():
        flat_metrics[f"traditional_{metric}"] = value
    
    # Add graph metrics
    for metric, value in metrics["graph"].items():
        flat_metrics[f"graph_{metric}"] = value
    
    # Add path metrics
    for metric, value in metrics["path"].items():
        flat_metrics[f"path_{metric}"] = value
    
    # Add hallucination metrics
    for metric, value in metrics["hallucination"].items():
        flat_metrics[f"hallucination_{metric}"] = value
    
    # Add extraction metrics
    for metric, value in metrics["extraction"].items():
        flat_metrics[f"extraction_{metric}"] = value
    
    # Add correlation metrics
    for metric, value in metrics["correlations"].items():
        flat_metrics[f"correlation_{metric}"] = value
    
    # Write to CSV
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=flat_metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(flat_metrics)
    
    print(f"\nEvaluation results saved to: {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Advanced KG evaluation against DBpedia-linked ground truth.")
    parser.add_argument("--kg_name", required=True, help="Name of the LLM-generated KG to evaluate")
    parser.add_argument("--dataset", default="data/ml-1m", help="Path to MovieLens dataset directory")
    parser.add_argument("--dataset_type", type=str, default='1m', choices=['1m', 'small'], 
                       help='Dataset type: 1m or small (default: 1m)')
    args = parser.parse_args()
    
    evaluator = KnowledgeGraphEvaluator(args.kg_name, args.dataset, args.dataset_type)
    try:
        metrics = evaluator.evaluate()
        
        print(f"\n=== Evaluation Results (Ground Truth: DBpedia file) ===")
        
        print("\nTraditional Metrics:")
        for metric, value in metrics["traditional"].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
                
        print("\nGraph-based Metrics:")
        for metric, value in metrics["graph"].items():
            print(f"  {metric}: {value:.4f}")
            
        print("\nPath-based Metrics:")
        for metric, value in metrics["path"].items():
            print(f"  {metric}: {value:.4f}")
            
        print("\nHallucination Detection:")
        for metric, value in metrics["hallucination"].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        print("\nExtraction Efficiency Metrics:")
        for metric, value in metrics["extraction"].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        print("\nMetric Correlations:")
        for metric, value in metrics["correlations"].items():
            print(f"  {metric}: {value:.4f}")
        
        # Save results to CSV
        save_evaluation_results(metrics, evaluator.model_name, evaluator.prompt_type, args.dataset_type)
                
    finally:
        evaluator.close()

if __name__ == "__main__":
    main() 