"""
Strategy 3 Enhanced: Advanced Graph-based Evaluation with Embeddings

This script evaluates LLM-generated knowledge graphs using the same metrics as str3
but replaces string-based fuzzy matching with embedding-based similarity.

Usage:
    python scripts/str3_enhanced_embedding_evaluate.py --kg_name "mistral_basic_kg"
"""

import argparse
from datetime import datetime
from pathlib import Path
import os
from tqdm import tqdm
import networkx as nx
from kg_evaluator_module import KnowledgeGraphEvaluator
import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple, Dict, Any, Set

# Relation mapping table: LLM phrase -> DBpedia ontology relation
RELATION_MAPPING = {
    'is a': 'type',
    'rdf:type': 'type',
    'stars': 'starring',
    'starring': 'starring',
    'directed by': 'director',
    'director': 'director',
    'released in': 'releasedate',
    'release date': 'releasedate',
    'release year': 'releasedate',
    'written by': 'writer',
    'writer': 'writer',
    'editing by': 'editing',
    'edited by': 'editing',
    'editing': 'editing',
    'based on': 'basedon',
    'music by': 'musiccomposer',
    'music': 'musiccomposer',
    'cinematography by': 'cinematography',
    'cinematography': 'cinematography',
    'produced by': 'producer',
    'producer': 'producer',
    'distributed by': 'distributor',
    'distributor': 'distributor',
    'language': 'language',
    'country': 'country',
    'runtime': 'runtime',
    'budget': 'budget',
    'gross': 'gross',
    'box office': 'gross',
}

class EnhancedEmbeddingEvaluator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the enhanced embedding evaluator."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        print(f"✓ Loaded sentence-transformer model: {model_name}")
    
    def triple_to_text(self, triple: Tuple[str, str, str]) -> str:
        """Convert triple to natural language text for embedding."""
        subject, predicate, obj = triple
        return f"{subject} {predicate} {obj}"
    
    def compute_embeddings(self, triples: List[Tuple[str, str, str]]) -> np.ndarray:
        """Compute embeddings for a list of triples."""
        texts = [self.triple_to_text(t) for t in triples]
        print(f"Computing embeddings for {len(texts)} triples...")
        
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            print(f"Error computing embeddings: {e}")
            # Use zero vectors as fallback
            return np.zeros((len(texts), 384))  # all-MiniLM-L6-v2 has 384 dimensions
    
    def compute_similarity_matrix(self, llm_triples: List[Tuple[str, str, str]], 
                                ref_triples: List[Tuple[str, str, str]]) -> np.ndarray:
        """Compute similarity matrix between LLM and reference triples."""
        # Compute embeddings
        print("Computing embeddings for LLM triples...")
        llm_embeddings = self.compute_embeddings(llm_triples)
        
        print("Computing embeddings for reference triples...")
        ref_embeddings = self.compute_embeddings(ref_triples)
        
        # Compute cosine similarity matrix
        print("Computing similarity matrix...")
        # Convert to torch tensors if needed
        if not isinstance(llm_embeddings, torch.Tensor):
            llm_embeddings = torch.tensor(llm_embeddings)
        if not isinstance(ref_embeddings, torch.Tensor):
            ref_embeddings = torch.tensor(ref_embeddings)
        
        similarity_matrix = util.pytorch_cos_sim(llm_embeddings, ref_embeddings).numpy()
        
        return similarity_matrix

def uri_to_name(uri):
    # Extract the last part of the URI (after last / or #)
    return re.split(r'[\/\#]', uri.strip())[-1]

def smart_normalize(text):
    # Lowercase, replace underscores/hyphens with spaces, remove punctuation, strip
    text = text.lower().replace('_', ' ').replace('-', ' ')
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

def normalize_triple(triple):
    # Accepts (subject, predicate, object) as URIs or plain text
    s, p, o = (smart_normalize(uri_to_name(x)) for x in triple)
    # Map predicate using RELATION_MAPPING if possible
    p_mapped = RELATION_MAPPING.get(p, p)
    return (s, p_mapped, o)

def load_triples_from_tsv(tsv_path):
    triples = set()
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                # Assume .tsv is subject, object, predicate (from sample)
                triple = (parts[0], parts[2], parts[1])  # (subject, predicate, object)
                triples.add(normalize_triple(triple))
    return triples

def build_graph(triples):
    G = nx.MultiDiGraph()
    for s, p, o in tqdm(triples, desc="Building graph"):
        G.add_edge(s, o, key=p)
    return G

def compute_graph_metrics(llm_graph, ref_graph):
    # Node and edge overlap
    llm_nodes = set(llm_graph.nodes())
    ref_nodes = set(ref_graph.nodes())
    node_overlap = len(llm_nodes & ref_nodes) / len(ref_nodes) if ref_nodes else 0.0
    llm_edges = set(llm_graph.edges(keys=True))
    ref_edges = set(ref_graph.edges(keys=True))
    edge_overlap = len(llm_edges & ref_edges) / len(ref_edges) if ref_edges else 0.0
    # Density
    llm_density = nx.density(llm_graph)
    ref_density = nx.density(ref_graph)
    # Component ratio
    llm_components = nx.number_weakly_connected_components(llm_graph)
    ref_components = nx.number_weakly_connected_components(ref_graph)
    component_ratio = llm_components / ref_components if ref_components else 0.0
    # Path overlap (shortest paths)
    llm_paths = set()
    ref_paths = set()
    for G, paths in [(llm_graph, llm_paths), (ref_graph, ref_paths)]:
        for source in tqdm(G.nodes(), desc="Finding shortest paths", leave=False):
            lengths = nx.single_source_shortest_path_length(G, source, cutoff=2)
            for target, l in lengths.items():
                if l > 0:
                    paths.add((source, target, l))
    path_overlap = len(llm_paths & ref_paths) / len(ref_paths) if ref_paths else 0.0
    return {
        "node_overlap": node_overlap,
        "edge_overlap": edge_overlap,
        "llm_density": llm_density,
        "ref_density": ref_density,
        "component_ratio": component_ratio,
        "path_overlap": path_overlap,
        "llm_nodes": len(llm_nodes),
        "ref_nodes": len(ref_nodes),
        "llm_edges": len(llm_edges),
        "ref_edges": len(ref_edges),
        "llm_components": llm_components,
        "ref_components": ref_components,
        "llm_paths": len(llm_paths),
        "ref_paths": len(ref_paths)
    }

def match_triples_with_embeddings(llm_triples, ref_triples, similarity_matrix, threshold=0.7):
    """Match triples using embedding similarity instead of fuzzy string matching."""
    
    match_stats = {'exact': 0, 'high_similarity': 0, 'medium_similarity': 0, 'low_similarity': 0, 'unmatched': 0}
    match_examples = {'high_similarity': [], 'medium_similarity': [], 'low_similarity': []}
    matched_triples = set()
    
    # Convert to lists for indexing
    llm_triples_list = list(llm_triples)
    ref_triples_list = list(ref_triples)
    
    for i, llm_triple in enumerate(llm_triples_list):
        # Get similarity scores for this triple
        similarities = similarity_matrix[i]
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        best_ref_triple = ref_triples_list[best_idx]
        
        # Classify match based on similarity
        if best_similarity >= 0.9:
            match_stats['exact'] += 1
            matched_triples.add(llm_triple)
        elif best_similarity >= 0.7:
            match_stats['high_similarity'] += 1
            matched_triples.add(llm_triple)
            if len(match_examples['high_similarity']) < 5:
                match_examples['high_similarity'].append((llm_triple, best_ref_triple, best_similarity))
        elif best_similarity >= 0.5:
            match_stats['medium_similarity'] += 1
            matched_triples.add(llm_triple)
            if len(match_examples['medium_similarity']) < 5:
                match_examples['medium_similarity'].append((llm_triple, best_ref_triple, best_similarity))
        elif best_similarity >= 0.3:
            match_stats['low_similarity'] += 1
            matched_triples.add(llm_triple)
            if len(match_examples['low_similarity']) < 5:
                match_examples['low_similarity'].append((llm_triple, best_ref_triple, best_similarity))
        else:
            match_stats['unmatched'] += 1
    
    return matched_triples, match_stats, match_examples

def main():
    parser = argparse.ArgumentParser(description="Strategy 3 Enhanced: Advanced graph-based evaluation with embeddings")
    parser.add_argument('--kg_name', required=True, help='Name of the LLM-generated KG in Neo4j')
    parser.add_argument('--ground_truth', choices=['dbpedia', 'golden_kg'], default='dbpedia',
                       help='Ground truth source to use for comparison (default: dbpedia)')
    parser.add_argument('--embedding_model', default='all-MiniLM-L6-v2', 
                       help='Sentence-transformer model to use (default: all-MiniLM-L6-v2)')
    parser.add_argument('--similarity_threshold', type=float, default=0.7, 
                       help='Similarity threshold for matching (default: 0.7)')
    args = parser.parse_args()

    print(f"\n[Strategy 3 Enhanced] Evaluating KG '{args.kg_name}' with embedding-based graph metrics...")
    print(f"Ground truth source: {args.ground_truth}")
    print(f"Using embedding model: {args.embedding_model}")
    print(f"Similarity threshold: {args.similarity_threshold}")
    
    # Initialize evaluators
    evaluator = KnowledgeGraphEvaluator(args.kg_name, dataset_dir='data/ml1m-fixed', dataset_type='1m')
    embedding_evaluator = EnhancedEmbeddingEvaluator(args.embedding_model)
    
    try:
        # Load LLM triples
        print("Fetching LLM triples from Neo4j...")
        llm_triples_raw = evaluator.get_triples(args.kg_name)
        llm_triples = set(normalize_triple(t) for t in llm_triples_raw)
        
        # Load ground truth based on choice
        if args.ground_truth == 'dbpedia':
            print("Loading DBpedia ground truth from TSV file...")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            tsv_path = os.path.join(script_dir, '..', 'data', 'ml1m_fixed', 'kg_triples_names.tsv')
            ref_triples_raw = load_triples_from_tsv(tsv_path)
            ref_triples = set(normalize_triple(t) for t in ref_triples_raw)
            ground_truth_name = "DBpedia"
        else:  # golden_kg
            print("Loading Golden KG ground truth from Neo4j...")
            ref_triples_raw = evaluator.get_triples('Golden KG (1m)')
            ref_triples = set(normalize_triple(t) for t in ref_triples_raw)
            ground_truth_name = "Golden KG (1m)"
        
        print(f"  LLM KG triples (raw): {len(llm_triples_raw)}")
        print(f"  LLM KG triples (normalized): {len(llm_triples)}")
        print(f"  {ground_truth_name} ground truth triples (normalized): {len(ref_triples)}")

        # Debug: print sample normalized triples
        print("Sample normalized LLM triples:", list(llm_triples)[:5])
        print(f"Sample normalized {ground_truth_name} triples:", list(ref_triples)[:5])

        # Convert to lists for embedding computation
        llm_triples_list = list(llm_triples)
        ref_triples_list = list(ref_triples)
        
        # Compute similarity matrix using embeddings
        print("\nComputing embedding-based similarity matrix...")
        similarity_matrix = embedding_evaluator.compute_similarity_matrix(llm_triples_list, ref_triples_list)
        
        # Match triples using embeddings
        print("Matching triples using embedding similarity...")
        matched_triples, match_stats, match_examples = match_triples_with_embeddings(
            llm_triples, ref_triples, similarity_matrix, args.similarity_threshold
        )
        
        # Calculate percentages for match stats
        total_triples = len(llm_triples)
        match_percentages = {
            'exact': (match_stats['exact'] / total_triples) * 100,
            'high_similarity': (match_stats['high_similarity'] / total_triples) * 100,
            'medium_similarity': (match_stats['medium_similarity'] / total_triples) * 100,
            'low_similarity': (match_stats['low_similarity'] / total_triples) * 100,
            'unmatched': (match_stats['unmatched'] / total_triples) * 100
        }
        
        print(f"Embedding match stats: {match_stats}")
        print(f"Match percentages:")
        print(f"  Exact matches: {match_stats['exact']} triples ({match_percentages['exact']:.1f}%)")
        print(f"  High similarity: {match_stats['high_similarity']} triples ({match_percentages['high_similarity']:.1f}%)")
        print(f"  Medium similarity: {match_stats['medium_similarity']} triples ({match_percentages['medium_similarity']:.1f}%)")
        print(f"  Low similarity: {match_stats['low_similarity']} triples ({match_percentages['low_similarity']:.1f}%)")
        print(f"  Unmatched: {match_stats['unmatched']} triples ({match_percentages['unmatched']:.1f}%)")
        print(f"  Total matched: {total_triples - match_stats['unmatched']} triples ({(100 - match_percentages['unmatched']):.1f}%)")
        
        for typ in ['high_similarity', 'medium_similarity', 'low_similarity']:
            if match_examples[typ]:
                print(f"Examples of {typ} matches:")
                for ex in match_examples[typ]:
                    print(f"  LLM: {ex[0]} <-> {ground_truth_name}: {ex[1]} (sim: {ex[2]:.3f})")

        # Use matched_triples for overlap, hallucination, etc.
        llm_graph = build_graph(matched_triples)
        ref_graph = build_graph(ref_triples)

        print("\nComputing advanced graph metrics...")
        graph_metrics = compute_graph_metrics(llm_graph, ref_graph)

        print("\nComputing hallucination rate...")
        hallucination_rate = 1 - (len(matched_triples) / len(llm_triples)) if llm_triples else 0.0
        hallucinated_count = len(llm_triples) - len(matched_triples)
        graph_metrics["hallucination_rate"] = hallucination_rate
        graph_metrics["hallucinated_triples"] = hallucinated_count
        
        # Calculate additional percentage metrics
        total_triples = len(llm_triples)
        total_ref_triples = len(ref_triples)
        matched_percentage = (len(matched_triples) / total_triples) * 100 if total_triples > 0 else 0.0
        coverage_percentage = (len(llm_triples) / total_ref_triples) * 100 if total_ref_triples > 0 else 0.0
        
        graph_metrics["matched_percentage"] = matched_percentage
        graph_metrics["coverage_percentage"] = coverage_percentage
        
        # Add embedding match stats to CSV
        for k, v in match_stats.items():
            graph_metrics[f"embedding_{k}_count"] = v
        
        # Add sample matches to CSV (as string)
        for typ in ['high_similarity', 'medium_similarity', 'low_similarity']:
            graph_metrics[f"embedding_{typ}_examples"] = str(match_examples[typ])
        
        # Add embedding model info
        graph_metrics["embedding_model"] = args.embedding_model
        graph_metrics["similarity_threshold"] = args.similarity_threshold

        model_name = evaluator.model_name
        prompt_type = evaluator.prompt_type
        ground_truth_suffix = "dbpedia" if args.ground_truth == 'dbpedia' else "golden_kg"
        out_path = Path('KG_evaluation') / f"evaluate_{model_name}_{prompt_type}_str3_enhanced_{ground_truth_suffix}.csv"
        os.makedirs('KG_evaluation', exist_ok=True)
        with open(out_path, 'w') as f:
            f.write('metric,value\n')
            for k, v in graph_metrics.items():
                f.write(f'{k},{v}\n')
        print(f"\nResults saved to {out_path}\n")
        print("Summary:")
        print("=== Graph Structure Metrics ===")
        print(f"  Node overlap: {graph_metrics['node_overlap']:.1%}")
        print(f"  Edge overlap: {graph_metrics['edge_overlap']:.1%}")
        print(f"  LLM density: {graph_metrics['llm_density']:.6f}")
        print(f"  Reference density: {graph_metrics['ref_density']:.6f}")
        print(f"  Component ratio: {graph_metrics['component_ratio']:.1%}")
        print(f"  Path overlap: {graph_metrics['path_overlap']:.1%}")
        
        print("\n=== Coverage Metrics ===")
        print(f"  LLM nodes: {graph_metrics['llm_nodes']:,}")
        print(f"  Reference nodes: {graph_metrics['ref_nodes']:,}")
        print(f"  LLM edges: {graph_metrics['llm_edges']:,}")
        print(f"  Reference edges: {graph_metrics['ref_edges']:,}")
        print(f"  Coverage percentage: {graph_metrics['coverage_percentage']:.1f}%")
        
        print("\n=== Quality Metrics ===")
        print(f"  Hallucination rate: {graph_metrics['hallucination_rate']:.1%}")
        print(f"  Matched percentage: {graph_metrics['matched_percentage']:.1f}%")
        print(f"  Exact matches: {graph_metrics['embedding_exact_count']} ({graph_metrics['embedding_exact_count']/total_triples*100:.1f}%)")
        print(f"  High similarity: {graph_metrics['embedding_high_similarity_count']} ({graph_metrics['embedding_high_similarity_count']/total_triples*100:.1f}%)")
        print(f"  Medium similarity: {graph_metrics['embedding_medium_similarity_count']} ({graph_metrics['embedding_medium_similarity_count']/total_triples*100:.1f}%)")
        print(f"  Low similarity: {graph_metrics['embedding_low_similarity_count']} ({graph_metrics['embedding_low_similarity_count']/total_triples*100:.1f}%)")
        print(f"  Unmatched: {graph_metrics['embedding_unmatched_count']} ({graph_metrics['embedding_unmatched_count']/total_triples*100:.1f}%)")
        
        print("\n=== Configuration ===")
        print(f"  Ground truth source: {args.ground_truth}")
        print(f"  Embedding model: {graph_metrics['embedding_model']}")
        print(f"  Similarity threshold: {graph_metrics['similarity_threshold']}")
    finally:
        evaluator.close()

if __name__ == '__main__':
    main() 