"""
Knowledge Graph Module using Neo4j
"""

from neo4j import GraphDatabase
from typing import List, Tuple, Optional

class KnowledgeGraph:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password123"):
        """Initialize the knowledge graph with Neo4j connection.
        
        Args:
            uri (str): Neo4j database URI
            user (str): Neo4j username
            password (str): Neo4j password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()
        
    def add_triple(self, subject: str, predicate: str, obj: str, confidence: float = 1.0):
        """Add a triple to the knowledge graph.
        
        Args:
            subject (str): Subject node
            predicate (str): Relationship type
            obj (str): Object node
            confidence (float): Confidence score of the relationship
        """
        with self.driver.session() as session:
            # Create or merge nodes and relationship
            query = """
            MERGE (s:Entity {name: $subject})
            MERGE (o:Entity {name: $object})
            MERGE (s)-[r:$predicate {confidence: $confidence}]->(o)
            """
            session.run(query, {
                "subject": subject,
                "object": obj,
                "predicate": predicate,
                "confidence": confidence
            })
            
    def query_triples(self, subject: Optional[str] = None, predicate: Optional[str] = None, obj: Optional[str] = None) -> List[Tuple[str, str, str, float]]:
        """Query triples from the knowledge graph.
        
        Args:
            subject (str, optional): Filter by subject
            predicate (str, optional): Filter by predicate
            obj (str, optional): Filter by object
            
        Returns:
            List[Tuple[str, str, str, float]]: List of (subject, predicate, object, confidence) tuples
        """
        with self.driver.session() as session:
            # Build the query based on provided filters
            query_parts = []
            params = {}
            
            if subject is not None:
                query_parts.append("s.name = $subject")
                params["subject"] = subject
            if predicate is not None:
                query_parts.append("type(r) = $predicate")
                params["predicate"] = predicate
            if obj is not None:
                query_parts.append("o.name = $object")
                params["object"] = obj
                
            where_clause = " AND ".join(query_parts) if query_parts else "1=1"
            
            query = f"""
            MATCH (s:Entity)-[r]->(o:Entity)
            WHERE {where_clause}
            RETURN s.name as subject, type(r) as predicate, o.name as object, r.confidence as confidence
            """
            
            result = session.run(query, params)
            return [(record["subject"], record["predicate"], record["object"], record["confidence"]) 
                   for record in result] 