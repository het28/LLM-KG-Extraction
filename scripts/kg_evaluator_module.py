import os
from neo4j import GraphDatabase
from pathlib import Path
from typing import Set, Tuple, List
import re

class KnowledgeGraphEvaluator:
    def __init__(self, kg_name, dataset_dir='data/ml1m-fixed', dataset_type='1m', neo4j_uri=None, neo4j_user=None, neo4j_password=None):
        self.kg_name = kg_name
        self.dataset_dir = dataset_dir
        self.dataset_type = dataset_type
        self.neo4j_uri = neo4j_uri or 'bolt://localhost:7687'
        self.neo4j_user = neo4j_user or 'neo4j'
        self.neo4j_password = neo4j_password or 'password123'
        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
        self.model_name, self.prompt_type = self._parse_kg_name(kg_name)

    def close(self):
        self.driver.close()

    def _parse_kg_name(self, kg_name):
        # Example: 'mistral_basic_kg' -> ('mistral', 'basic')
        parts = kg_name.lower().split('_')
        model = parts[0] if len(parts) > 0 else 'unknown'
        prompt = parts[1] if len(parts) > 1 else 'unknown'
        return model, prompt

    def get_triples(self, kg_name) -> Set[Tuple[str, str, str]]:
        with self.driver.session() as session:
            result = session.run(f"""
                MATCH (s)-[p]->(o) WHERE p.kg_name = $kg_name
                RETURN s.name AS s, type(p) AS p, o.name AS o
            """, kg_name=kg_name)
            triples = set()
            for record in result:
                s, p, o = record['s'], record['p'], record['o']
                triples.add((s, p, o))
            return triples

    def compute_traditional_metrics(self, llm_triples: Set[Tuple[str, str, str]], ref_triples: Set[Tuple[str, str, str]], use_soft_matching=False, sim_func=None, threshold=0.8):
        if not use_soft_matching:
            true_positives = llm_triples & ref_triples
            false_positives = llm_triples - ref_triples
            false_negatives = ref_triples - llm_triples
            precision = len(true_positives) / len(llm_triples) if llm_triples else 0.0
            recall = len(true_positives) / len(ref_triples) if ref_triples else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            return {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positives": len(true_positives),
                "false_positives": len(false_positives),
                "false_negatives": len(false_negatives),
                "llm_triples": len(llm_triples),
                "ref_triples": len(ref_triples)
            }
        else:
            # Soft matching using sim_func (e.g., Sentence-BERT cosine similarity)
            llm_list = list(llm_triples)
            ref_list = list(ref_triples)
            matched_ref = set()
            soft_true_positives = 0
            total_similarity = 0.0
            for i, t1 in enumerate(llm_list):
                best_sim = 0.0
                best_j = -1
                for j, t2 in enumerate(ref_list):
                    sim = sim_func(t1, t2)
                    if sim > best_sim:
                        best_sim = sim
                        best_j = j
                if best_sim >= threshold:
                    soft_true_positives += 1
                    matched_ref.add(best_j)
                    total_similarity += best_sim
            soft_false_positives = len(llm_list) - soft_true_positives
            soft_false_negatives = len(ref_list) - len(matched_ref)
            soft_precision = soft_true_positives / len(llm_list) if llm_list else 0.0
            soft_recall = soft_true_positives / len(ref_list) if ref_list else 0.0
            soft_f1 = 2 * soft_precision * soft_recall / (soft_precision + soft_recall) if (soft_precision + soft_recall) > 0 else 0.0
            avg_similarity = total_similarity / soft_true_positives if soft_true_positives > 0 else 0.0
            return {
                "soft_precision": soft_precision,
                "soft_recall": soft_recall,
                "soft_f1": soft_f1,
                "avg_similarity": avg_similarity,
                "soft_true_positives": soft_true_positives,
                "soft_false_positives": soft_false_positives,
                "soft_false_negatives": soft_false_negatives,
                "llm_triples": len(llm_list),
                "ref_triples": len(ref_list)
            } 