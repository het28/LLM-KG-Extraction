import argparse
import csv
from neo4j import GraphDatabase
import random
from pathlib import Path
import pandas as pd
import pickle

# ---- CONFIGURE THESE ----
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123" 
# ------------------------

def get_structural_metrics(driver, kg_name):
    with driver.session() as session:
        node_count = session.run(
            "MATCH (n) WHERE n.kg_name = $kg_name RETURN count(n) as count",
            kg_name=kg_name
        ).single()["count"]
        rel_count = session.run(
            "MATCH ()-[r]->() WHERE r.kg_name = $kg_name RETURN count(r) as count",
            kg_name=kg_name
        ).single()["count"]
        labels = session.run(
            "MATCH (n) WHERE n.kg_name = $kg_name RETURN DISTINCT labels(n) as labels",
            kg_name=kg_name
        )
        rel_types = session.run(
            "MATCH ()-[r]->() WHERE r.kg_name = $kg_name RETURN DISTINCT type(r) as type",
            kg_name=kg_name
        )
        print(f"Nodes: {node_count}")
        print(f"Relationships: {rel_count}")
        print(f"Node labels: {[row['labels'] for row in labels]}")
        print(f"Relationship types: {[row['type'] for row in rel_types]}")

def sample_triples(driver, kg_name, sample_size):
    with driver.session() as session:
        result = session.run(
            "MATCH (s)-[p]->(o) WHERE p.kg_name = $kg_name RETURN s, type(p) as rel, o LIMIT 1000000",
            kg_name=kg_name
        )
        triples = [(r["s"], r["rel"], r["o"]) for r in result]
        if not triples:
            print("No triples found for this KG name.")
            return []
        sampled = random.sample(triples, min(sample_size, len(triples)))
        formatted = []
        for s, rel, o in sampled:
            formatted.append({
                "subject": s.get("name", str(s.id)),
                "predicate": rel,
                "object": o.get("name", str(o.id))
            })
        return formatted

def export_samples(samples, output_path):
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["subject", "predicate", "object"])
        writer.writeheader()
        for row in samples:
            writer.writerow(row)
    print(f"Exported {len(samples)} samples to {output_path}")

def get_llm_kg_triples(driver, kg_name, eval_type):
    with driver.session() as session:
        if eval_type == "genre":
            cypher = "MATCH (s)-[p:HAS_GENRE]->(o) WHERE p.kg_name = $kg_name RETURN s.name AS s, 'HAS_GENRE' AS p, o.name AS o"
        elif eval_type == "tag":
            cypher = "MATCH (s)-[p:HAS_TAG]->(o) WHERE p.kg_name = $kg_name RETURN s.name AS s, 'HAS_TAG' AS p, o.name AS o"
        else:
            cypher = "MATCH (s)-[p]->(o) WHERE p.kg_name = $kg_name RETURN s.name AS s, type(p) AS p, o.name AS o"
        result = session.run(cypher, kg_name=kg_name)
        triples = set()
        for r in result:
            triples.add((str(r["s"]).strip(), str(r["p"]).strip(), str(r["o"]).strip()))
        return triples

def get_neo4j_golden_kg_triples(driver, eval_type):
    with driver.session() as session:
        if eval_type == "genre":
            cypher = "MATCH (s)-[p:HAS_GENRE]->(o) WHERE p.kg_name = 'Golden KG' RETURN s.name AS s, 'HAS_GENRE' AS p, o.name AS o"
        elif eval_type == "tag":
            cypher = "MATCH (s)-[p:HAS_TAG]->(o) WHERE p.kg_name = 'Golden KG' RETURN s.name AS s, 'HAS_TAG' AS p, o.name AS o"
        else:
            cypher = "MATCH (s)-[p]->(o) WHERE p.kg_name = 'Golden KG' RETURN s.name AS s, type(p) AS p, o.name AS o"
        result = session.run(cypher)
        triples = set()
        for r in result:
            triples.add((str(r["s"]).strip(), str(r["p"]).strip(), str(r["o"]).strip()))
        return triples

def evaluate_kg(llm_triples, ref_triples):
    true_positives = llm_triples & ref_triples
    false_positives = llm_triples - ref_triples
    false_negatives = ref_triples - llm_triples
    precision = len(true_positives) / len(llm_triples) if llm_triples else 0.0
    recall = len(true_positives) / len(ref_triples) if ref_triples else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print("\n--- Automatic Evaluation Against Reference KG ---")
    print(f"True Positives: {len(true_positives)}")
    print(f"False Positives: {len(false_positives)}")
    print(f"False Negatives: {len(false_negatives)}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    return precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description="Evaluate a Knowledge Graph in Neo4j.")
    parser.add_argument("--kg_name", required=True, help="KG name/model+prompt identifier")
    parser.add_argument("--sample_size", type=int, default=1000000, help="Number of triples to sample (set very large to get all)")
    parser.add_argument("--output_samples", type=str, help="Path to export sampled triples as CSV")
    parser.add_argument("--structural_metrics", action="store_true", help="Print structural metrics")
    parser.add_argument("--eval_type", type=str, choices=["genre", "tag", "all"], default="all", help="Type of triples to evaluate (genre, tag, all)")
    args = parser.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    if args.structural_metrics:
        print("--- Structural Metrics ---")
        get_structural_metrics(driver, args.kg_name)
        print()

    print(f"--- Sampling {args.sample_size} triples from KG '{args.kg_name}' ---")
    samples = sample_triples(driver, args.kg_name, args.sample_size)
    for i, triple in enumerate(samples, 1):
        print(f"{i}. ({triple['subject']}) -[{triple['predicate']}] -> ({triple['object']})")

    if args.output_samples:
        export_samples(samples, args.output_samples)

    print(f"\n--- Loading Reference KG from Neo4j (Golden KG) ---")
    ref_triples = get_neo4j_golden_kg_triples(driver, args.eval_type)
    llm_triples = get_llm_kg_triples(driver, args.kg_name, args.eval_type)
    evaluate_kg(llm_triples, ref_triples)

    driver.close()

if __name__ == "__main__":
    main() 