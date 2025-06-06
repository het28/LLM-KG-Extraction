"""
Example script demonstrating LLM-based knowledge graph extraction.
"""

from kg_extractor import KnowledgeGraphExtractor
from tqdm import tqdm

def main():
    # Initialize the knowledge graph extractor
    extractor = KnowledgeGraphExtractor()
    
    try:
        # Example texts to process
        texts = [
            """Star Wars is a science fiction movie directed by George Lucas. 
            It was released in 1977 and stars Mark Hamill as Luke Skywalker, 
            Harrison Ford as Han Solo, and Carrie Fisher as Princess Leia.""",
            
            """The Matrix is a 1999 science fiction film directed by the Wachowskis. 
            It stars Keanu Reeves as Neo, Laurence Fishburne as Morpheus, 
            and Carrie-Anne Moss as Trinity. The movie explores themes of reality and consciousness."""
        ]
        
        # Process each text
        for text in texts:
            print(f"\nProcessing text: {text[:100]}...")
            
            # Extract triples using LLM
            triples = extractor.extract_triples(text)
            
            # Add triples to the knowledge graph
            print(f"Extracted {len(triples)} triples:")
            for subject, predicate, obj in triples:
                print(f"- ({subject}, {predicate}, {obj})")
                extractor.add_triple(subject, predicate, obj)
        
        # Example queries
        print("\nExample Queries:")
        
        # 1. Find all movies
        print("\n1. All movies:")
        movies = extractor.query_triples(predicate="is_a")
        for subject, _, obj, _ in movies:
            print(f"- {subject} is a {obj}")
        
        # 2. Find all directors
        print("\n2. All directors:")
        directors = extractor.query_triples(predicate="directed_by")
        for movie, _, director, _ in directors:
            print(f"- {movie} was directed by {director}")
        
        # 3. Find all actors
        print("\n3. All actors:")
        actors = extractor.query_triples(predicate="stars")
        for movie, _, actor, _ in actors:
            print(f"- {movie} stars {actor}")
            
    finally:
        # Always close the Neo4j connection
        extractor.close()

if __name__ == "__main__":
    main() 