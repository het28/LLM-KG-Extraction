"""
Build a reference/golden knowledge graph from MovieLens data using classic NLP tools.

Usage:
    python src/golden_kg_builder.py --dataset data/ml-latest-small --max_samples 0

- --dataset: Path to the dataset directory (containing movies.csv and tags.csv)
- --max_samples: Number of samples to process (0 = all)
"""

import argparse
import pandas as pd
from pathlib import Path
import sys
import spacy
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import numpy as np
import re
import csv

# Neo4j connection details - Update these with your Neo4j credentials
NEO4J_URI = ""  # Change this if your Neo4j is not running locally
NEO4J_USER = "neo4j"                 # Your Neo4j username
NEO4J_PASSWORD = "password"     # Your Neo4j password

def load_movielens_small(dataset_dir, max_samples=0):
    movies_path = Path(dataset_dir) / "movies.csv"
    tags_path = Path(dataset_dir) / "tags.csv"
    if not movies_path.exists() or not tags_path.exists():
        print(f"Error: movies.csv or tags.csv not found in {dataset_dir}")
        sys.exit(1)
    movies_df = pd.read_csv(movies_path)
    tags_df = pd.read_csv(tags_path)
    movie_texts = []
    for _, movie in movies_df.iterrows():
        movie_tags = tags_df[tags_df['movieId'] == movie['movieId']]['tag'].tolist()
        text = f"{movie['title']} ({movie['title'].split('(')[-1].strip(')')}) is a {movie['genres'].replace('|', ', ')} film."
        if movie_tags:
            text += f" Tags: {', '.join(movie_tags)}."
        movie_texts.append(text)
    if max_samples and max_samples > 0:
        return movie_texts[:max_samples], movies_df.to_dict(orient='records')
    return movie_texts, movies_df.to_dict(orient='records')

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
        return movie_texts[:max_samples], movies_df.to_dict(orient='records')
    return movie_texts, movies_df.to_dict(orient='records')

def extract_entities(text, nlp, movie_id=None):
    """Extract entities from text using spaCy NER with improved entity handling."""
    doc = nlp(text)
    entities = []
    
    # Extract named entities
    for ent in doc.ents:
        # Clean and normalize entity text
        entity_text = ent.text.strip()
        if entity_text:
            entities.append((entity_text, ent.label_))
    
    # Extract movie titles (they might not be caught by NER)
    # Look for patterns like "Movie Name (Year)"
    movie_pattern = r'([^(]+)\s*\((\d{4})\)'
    movie_matches = re.finditer(movie_pattern, text)
    for match in movie_matches:
        movie_title = match.group(1).strip()
        year = match.group(2)
        # Make movie title unique by including movie_id if available
        if movie_id:
            movie_title = f"{movie_title} (ID: {movie_id})"
        entities.append((movie_title, "MOVIE"))
        entities.append((year, "YEAR"))
    
    return entities

def normalize_entity(entity, model, entity_list):
    """Normalize entity using Sentence-BERT with improved matching."""
    # Clean entity text
    entity = entity.strip().lower()
    
    # Direct match first
    if entity in [e.lower() for e in entity_list]:
        return entity
    
    # If no direct match, use semantic similarity
    entity_embedding = model.encode(entity)
    entity_list_embeddings = model.encode(entity_list)
    
    # Compute cosine similarity
    similarities = np.dot(entity_list_embeddings, entity_embedding) / (
        np.linalg.norm(entity_list_embeddings, axis=1) * 
        np.linalg.norm(entity_embedding)
    )
    
    # Get best match
    best_match_idx = np.argmax(similarities)
    best_similarity = similarities[best_match_idx]
    
    # Only return match if similarity is above threshold
    if best_similarity > 0.7:  # Adjust threshold as needed
        return entity_list[best_match_idx]
    return None

def extract_relations(entities, text, movie_data, movie_id=None):
    """Extract relations using improved template rules and movie data."""
    relations = []
    
    # Create lookup dictionaries for faster access
    movie_titles = {m['title'].lower(): m for m in movie_data}
    genres = set()
    for m in movie_data:
        genres.update(m['genres'].split('|'))
    
    # Process each entity pair
    for i, (subj, subj_type) in enumerate(entities):
        for j, (obj, obj_type) in enumerate(entities):
            if i != j:
                # Movie-Genre relations
                if subj_type == "MOVIE" and obj in genres:
                    relations.append((subj, "HAS_GENRE", obj))
                
                # Movie-Year relations
                elif subj_type == "MOVIE" and obj_type == "YEAR":
                    relations.append((subj, "RELEASED_IN", obj))
                
                # Movie-Tag relations (if tags are in entities)
                elif subj_type == "MOVIE" and "tag" in obj.lower():
                    relations.append((subj, "HAS_TAG", obj))
    
    return relations

def store_triples_in_neo4j(triples, dataset_type='small'):
    """Store triples in Neo4j with improved error handling and batch processing."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    # Set KG name based on dataset type
    kg_name = f"Golden KG ({dataset_type})"
    
    try:
        with driver.session() as session:
            # Clear existing data for this KG
            print(f"Clearing existing data for {kg_name}...")
            session.run("""
                MATCH (n:Entity {kg_name: $kg_name})
                DETACH DELETE n
            """, kg_name=kg_name)
            
            # Batch process triples
            batch_size = 1000
            for i in range(0, len(triples), batch_size):
                batch = triples[i:i + batch_size]
                
                # Create a single query for the batch
                query = """
                UNWIND $batch AS triple
                MERGE (s:Entity {name: triple.subject, kg_name: $kg_name})
                MERGE (o:Entity {name: triple.object, kg_name: $kg_name})
                MERGE (s)-[r:`${triple.predicate}` {kg_name: $kg_name}]->(o)
                """
                
                # Execute batch
                session.run(query, batch=[{
                    'subject': s,
                    'predicate': p,
                    'object': o
                } for s, p, o in batch], kg_name=kg_name)
                
    except Exception as e:
        print(f"Error storing triples: {e}")
    finally:
        driver.close()

def save_triples_to_csv(triples, dataset_type='small', output_path=None):
    if output_path is None:
        output_path = f"golden_kg_triples_{dataset_type}.csv"
    with open(output_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "predicate", "object"])
        for s, p, o in triples:
            writer.writerow([s, p, o])
    print(f"Saved {len(triples)} triples to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Build a reference/golden KG from MovieLens data.")
    parser.add_argument('--dataset', type=str, default='data/ml-1m', help='Path to dataset directory (default: data/ml-1m)')
    parser.add_argument('--dataset_type', type=str, default='1m', choices=['1m'], help='Dataset type: 1m (default: 1m)')
    parser.add_argument('--max_samples', type=int, default=0, help='Number of samples to process (0 = all)')
    args = parser.parse_args()
    print(f"Loading data from {args.dataset} ...")
    movie_texts, movie_data = load_movielens_1m(args.dataset, args.max_samples)
    print(f"Loaded {len(movie_texts)} movie descriptions.")

    # Load spaCy NER model
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    # Load Sentence-BERT model
    print("Loading Sentence-BERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Extract entities and relations
    print("Extracting entities and relations...")
    all_triples = []
    for idx, (text, movie) in enumerate(zip(movie_texts, movie_data)):
        entities = extract_entities(text, nlp, movie['movieId'])
        relations = extract_relations(entities, text, movie_data, movie['movieId'])
        all_triples.extend(relations)

    # Store triples in Neo4j
    print(f"Storing {len(all_triples)} triples in Neo4j...")
    store_triples_in_neo4j(all_triples, args.dataset_type)
    print("Done! Check your Neo4j database for the reference knowledge graph.")

    # Save triples to CSV for local import
    save_triples_to_csv(all_triples, args.dataset_type)

if __name__ == "__main__":
    main()
