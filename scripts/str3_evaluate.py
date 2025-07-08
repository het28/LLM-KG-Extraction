import argparse
from datetime import datetime
from pathlib import Path
import os
from tqdm import tqdm
import networkx as nx
from kg_evaluator_module import KnowledgeGraphEvaluator
import re
import difflib

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

def uri_to_name(uri):
    # Extract the last part of the URI (after last / or #)
    return re.split(r'[\/\#]', uri.strip())[-1]

def smart_normalize(text):
    # Lowercase, replace underscores/hyphens with spaces, remove punctuation, strip
    text = text.lower().replace('_', ' ').replace('-', ' ')
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

def fuzzy_match(target, candidates, threshold=0.85):
    # Try exact match first
    if target in candidates:
        return target, 'exact'
    # Fuzzy match (Levenshtein/difflib)
    match = difflib.get_close_matches(target, candidates, n=1, cutoff=threshold)
    if match:
        return match[0], 'fuzzy'
    # Substring match (for relations like 'star' in 'starring')
    for cand in candidates:
        if target in cand or cand in target:
            return cand, 'postfix/prefix'
    # Underscore/hyphen difference (already handled by smart_normalize, but log if only difference)
    for cand in candidates:
        if target.replace(' ', '') == cand.replace(' ', ''):
            return cand, 'underscore/hyphen'
    return None, None

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

def match_triples_fuzzy(llm_triples, ref_triples):
    # Build index for ref_triples by predicate for efficiency
    ref_by_pred = {}
    for s, p, o in ref_triples:
        ref_by_pred.setdefault(p, []).append((s, o))
    
    match_stats = {'exact': 0, 'fuzzy': 0, 'postfix/prefix': 0, 'underscore/hyphen': 0, 'unmatched': 0}
    match_examples = {'fuzzy': [], 'postfix/prefix': [], 'underscore/hyphen': []}
    matched_triples = set()
    for s1, p1, o1 in llm_triples:
        # Fuzzy match predicate
        pred_candidates = list(ref_by_pred.keys())
        p2, pred_type = fuzzy_match(p1, pred_candidates)
        if p2:
            # Fuzzy match subject
            subj_candidates = [s for s, _ in ref_by_pred[p2]]
            s2, subj_type = fuzzy_match(s1, subj_candidates)
            # Fuzzy match object
            obj_candidates = [o for ss, o in ref_by_pred[p2] if ss == s2] if s2 else []
            o2, obj_type = fuzzy_match(o1, obj_candidates)
            if s2 and o2:
                # Count the most significant fuzz type
                if pred_type == 'exact' and subj_type == 'exact' and obj_type == 'exact':
                    match_stats['exact'] += 1
                else:
                    for t, typ in zip(['predicate', 'subject', 'object'], [pred_type, subj_type, obj_type]):
                        if typ and typ != 'exact':
                            match_stats[typ] += 1
                            if len(match_examples[typ]) < 5:
                                match_examples[typ].append(((s1, p1, o1), (s2, p2, o2), t))
                matched_triples.add((s1, p1, o1))
                continue
        match_stats['unmatched'] += 1
    return matched_triples, match_stats, match_examples

def main():
    parser = argparse.ArgumentParser(description="Strategy 3: Advanced graph-based and hallucination metrics evaluation")
    parser.add_argument('--kg_name', required=True, help='Name of the LLM-generated KG in Neo4j')
    args = parser.parse_args()

    print(f"\n[Strategy 3] Evaluating KG '{args.kg_name}' with advanced graph-based and hallucination metrics...")
    evaluator = KnowledgeGraphEvaluator(args.kg_name, dataset_dir='data/ml1m-fixed', dataset_type='1m')
    try:
        print("Fetching triples from Neo4j and DBpedia ground truth TSV file...")
        llm_triples_raw = evaluator.get_triples(args.kg_name)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tsv_path = os.path.join(script_dir, '..', 'data', 'ml1m_fixed', 'kg_triples_names.tsv')
        ref_triples = load_triples_from_tsv(tsv_path)
        # Normalize LLM triples
        llm_triples = set(normalize_triple(t) for t in llm_triples_raw)
        ref_triples = set(normalize_triple(t) for t in ref_triples)
        print(f"  LLM KG triples (raw): {len(llm_triples_raw)}")
        print(f"  LLM KG triples (normalized): {len(llm_triples)}")
        print(f"  DBpedia ground truth triples (from TSV, normalized): {len(ref_triples)}")

        # Debug: print sample normalized triples
        print("Sample normalized LLM triples:", list(llm_triples)[:5])
        print("Sample normalized DBpedia triples:", list(ref_triples)[:5])

        # Fuzzy matching
        matched_triples, match_stats, match_examples = match_triples_fuzzy(llm_triples, ref_triples)
        print(f"Fuzzy match stats: {match_stats}")
        for typ in ['fuzzy', 'postfix/prefix', 'underscore/hyphen']:
            if match_examples[typ]:
                print(f"Examples of {typ} matches:")
                for ex in match_examples[typ]:
                    print(f"  LLM: {ex[0]} <-> DBpedia: {ex[1]} (on {ex[2]})")

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
        # Add fuzzy match stats to CSV
        for k, v in match_stats.items():
            graph_metrics[f"fuzzy_{k}_count"] = v
        # Add sample matches to CSV (as string)
        for typ in ['fuzzy', 'postfix/prefix', 'underscore/hyphen']:
            graph_metrics[f"fuzzy_{typ}_examples"] = str(match_examples[typ])

        model_name = evaluator.model_name
        prompt_type = evaluator.prompt_type
        out_path = Path('KG_evaluation') / f"evaluate_{model_name}_{prompt_type}_str3.csv"
        os.makedirs('KG_evaluation', exist_ok=True)
        with open(out_path, 'w') as f:
            f.write('metric,value\n')
            for k, v in graph_metrics.items():
                f.write(f'{k},{v}\n')
        print(f"\nResults saved to {out_path}\n")
        print("Summary:")
        for k, v in graph_metrics.items():
            print(f"  {k}: {v}")
    finally:
        evaluator.close()

if __name__ == '__main__':
    main() 