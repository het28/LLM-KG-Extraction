"""
Advanced Knowledge Graph Extractor using LLM with Self-Consistency Voting and KG Evolution Loop
"""

import os
import time
import csv
from datetime import datetime
from typing import List, Tuple, Dict, Any
from neo4j import GraphDatabase
from langchain_community.llms import Ollama
from langchain_together import Together
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from collections import defaultdict
import random

class AdvancedKnowledgeGraphExtractor:
    def __init__(self, neo4j_uri: str = "neo4j_url", 
                 neo4j_user: str = "neo4j", 
                 neo4j_password: str = "password",
                 model_name: str = "mistral",
                 prompt_type: str = "basic",
                 num_iterations: int = 3,
                 voting_threshold: float = 0.7,
                 #together_api_key: str = "",
                 dataset_type: str = None):
        """Initialize the advanced knowledge graph extractor.
        
        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            model_name: Name of the LLM to use
            prompt_type: Type of prompt to use
            num_iterations: Number of extraction iterations for self-consistency
            voting_threshold: Minimum agreement threshold for triple acceptance
            dataset_type: Dataset type for KG naming (e.g., '1m')
        """
        # Initialize Neo4j connection
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        # Set KG name with dataset type suffix if provided
        if dataset_type == '1m':
            self.kg_name = f"advance_{model_name}_{prompt_type}_kg (1m)"
        else:
            self.kg_name = f"advance_{model_name}_{prompt_type}_kg"
        
        # Initialize LLM
        # Model selection logic
        if model_name in ["mistral", "llama2", "qwen", "gemma", "deepseek"]:
            # Use Ollama for all these models
            ollama_model_map = {
                "mistral": "mistral",
                "llama2": "llama2:latest",
                "qwen": "qwen3:8b",
                "gemma": "gemma3:12b",
                "deepseek": "deepseek-r1:8b",
                "granite": "granite3.3:8b"
            }
            self.llm = Ollama(model=ollama_model_map[model_name])
        else:
            raise ValueError(f"Unknown or unsupported model: {model_name}")
        
        # Store parameters
        self.model_name = model_name
        self.prompt_type = prompt_type
        self.num_iterations = num_iterations
        self.voting_threshold = voting_threshold
        
        # Create extraction prompts (reuse existing prompts from kg_extractor.py)
        self.prompt = self._create_prompt(prompt_type)
        
        # Create LLM chain
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        
    def _create_prompt(self, prompt_type: str) -> PromptTemplate:
        """Create the appropriate prompt template."""
        # Create all prompt templates
        self.basic_prompt = PromptTemplate(
            input_variables=["text"],
            template="""You are a knowledge graph extraction expert. Extract factual triples from the following text.

Entity Types:
- Movie: Film titles and their identifiers
- Person: Directors, actors, writers, and other people
- Genre: Movie categories (e.g., Action, Drama, Comedy)
- Year: Release years in YYYY format
- Tag: User-provided descriptive tags

Relationship Types:
- is_a: Categorization (e.g., "The Matrix is_a film")
- has_genre: Movie genres (e.g., "The Matrix has_genre Science Fiction")
- released_in: Release years (e.g., "The Matrix released_in 1999")
- directed_by: Directors (e.g., "The Matrix directed_by Wachowskis")
- stars: Main actors (e.g., "The Matrix stars Keanu Reeves")
- plays_character: Character roles (e.g., "Keanu Reeves plays_character Neo")
- has_tag: User tags (e.g., "The Matrix has_tag mind-bending")

Guidelines:
1. Extract only factual information, no opinions
2. Use consistent entity names across triples
3. Include confidence score (0.0-1.0) for each triple
4. Format: (subject, predicate, object, confidence)

Example:
Input: "The Matrix (1999) is a science fiction film directed by the Wachowskis, starring Keanu Reeves as Neo."
Output:
(The Matrix, is_a, film, 1.0)
(The Matrix, has_genre, Science Fiction, 1.0)
(The Matrix, released_in, 1999, 1.0)
(The Matrix, directed_by, Wachowskis, 1.0)
(The Matrix, stars, Keanu Reeves, 1.0)
(Keanu Reeves, plays_character, Neo, 1.0)

Now extract knowledge triples from this text:
{text}

Triples:"""
        )
        
        self.complex_prompt = PromptTemplate(
            input_variables=["text"],
            template="""You are a knowledge graph extraction expert. Extract structured knowledge from the given text about movies and related entities.

Entity Definitions:
- Movie: Film titles, including series and sequels
- Person: Directors, actors, writers, producers, and other crew
- Genre: Movie categories and sub-genres
- Year: Release years and production years
- Tag: User-provided descriptive tags and keywords
- Character: Fictional characters in movies
- Plot: Key plot elements and story arcs

Relationship Types and Constraints:
1. Movie Attributes:
   - is_a: (Movie → Genre/Type) [1:1]
   - has_genre: (Movie → Genre) [1:N]
   - released_in: (Movie → Year) [1:1]
   - has_plot: (Movie → Plot) [1:N]

2. Movie Relationships:
   - is_sequel_to: (Movie → Movie) [1:1]
   - is_remake_of: (Movie → Movie) [1:1]
   - is_adaptation_of: (Movie → Source) [1:1]

3. Person Relationships:
   - directed_by: (Movie → Person) [1:N]
   - stars: (Movie → Person) [1:N]
   - plays_character: (Person → Character) [1:N]

4. Tag Relationships:
   - has_tag: (Movie → Tag) [1:N]

Validation Rules:
1. Entity Consistency:
   - Use same names for same entities
   - Maintain proper capitalization
   - Include years in movie titles

2. Relationship Validation:
   - Check temporal consistency
   - Verify entity existence
   - Validate relationship cardinality

3. Confidence Scoring:
   - 1.0: Explicitly stated in text
   - 0.8: Strongly implied
   - 0.6: Moderately implied
   - 0.4: Weakly implied
   - 0.2: Speculative

Example Input:
"The Matrix (1999) is a science fiction film directed by the Wachowskis, starring Keanu Reeves as Neo. It was followed by The Matrix Reloaded in 2003."

Example Output:
(The Matrix, is_a, film, 1.0)
(The Matrix, has_genre, Science Fiction, 1.0)
(The Matrix, released_in, 1999, 1.0)
(The Matrix, directed_by, Wachowskis, 1.0)
(The Matrix, stars, Keanu Reeves, 1.0)
(Keanu Reeves, plays_character, Neo, 1.0)
(The Matrix Reloaded, is_sequel_to, The Matrix, 1.0)
(The Matrix Reloaded, released_in, 2003, 1.0)

Now extract knowledge triples from this text:
{text}
 
Triples:"""
        )

        self.conversation_prompt = PromptTemplate(
            input_variables=["text"],
            template="""You are a knowledge graph extraction expert analyzing movie descriptions. Extract knowledge triples while maintaining context across multiple sentences.

Entity Tracking:
1. Entity Resolution:
   - Track entities across sentences
   - Resolve pronouns and references
   - Maintain consistent naming
   - Handle implicit entities

2. Temporal Context:
   - Track timeline of events
   - Handle flashbacks and flash-forwards
   - Maintain chronological order
   - Resolve temporal ambiguities

3. Narrative Context:
   - Track plot progression
   - Maintain character arcs
   - Handle subplots
   - Preserve story context

Relationship Types:
1. Direct Relationships:
   - is_a: Entity categorization
   - has_genre: Movie genres
   - released_in: Release years
   - directed_by: Directors
   - stars: Main actors
   - plays_character: Character roles
   - has_tag: User tags

2. Implicit Relationships:
   - is_sequel_to: Movie sequels
   - is_remake_of: Movie remakes
   - is_adaptation_of: Book adaptations
   - has_plot: Plot elements

Validation Steps:
1. Entity Validation:
   - Check entity consistency
   - Verify entity existence
   - Resolve ambiguities
   - Track entity changes

2. Relationship Validation:
   - Verify temporal consistency
   - Check relationship validity
   - Validate implicit relationships
   - Score relationship confidence

3. Context Validation:
   - Verify narrative consistency
   - Check temporal coherence
   - Validate plot progression
   - Score context confidence

Example Input:
"The Matrix (1999) is a science fiction film. It was directed by the Wachowskis. The movie stars Keanu Reeves as Neo, a computer programmer. The story follows Neo as he discovers the true nature of reality."

Example Output:
(The Matrix, is_a, film, 1.0)
(The Matrix, has_genre, Science Fiction, 1.0)
(The Matrix, released_in, 1999, 1.0)
(The Matrix, directed_by, Wachowskis, 1.0)
(The Matrix, stars, Keanu Reeves, 1.0)
(Keanu Reeves, plays_character, Neo, 1.0)
(Neo, is_a, computer programmer, 1.0)
(The Matrix, has_plot, story about discovering true nature of reality, 0.8)

Now extract knowledge triples from this text, maintaining context across sentences:
{text}

Triples:"""
        )

        self.structured_prompt = PromptTemplate(
            input_variables=["text"],
            template="""You are a knowledge graph extraction expert. Extract structured knowledge triples following a strict format and relationship types.

Entity Type Definitions:
1. Movie:
   - Title: Full movie title with year
   - Type: Film, series, sequel, etc.
   - Attributes: Genre, year, plot

2. Person:
   - Name: Full name
   - Role: Director, actor, writer, etc.
   - Character: If applicable

3. Genre:
   - Name: Genre name
   - Type: Main genre or sub-genre

4. Year:
   - Format: YYYY
   - Type: Release year, production year

5. Tag:
   - Name: User-provided tag
   - Type: Descriptive, thematic, etc.

Relationship Type Definitions:
1. Categorization:
   - is_a: (Entity → Type) [1:1]
   - has_genre: (Movie → Genre) [1:N]

2. Temporal:
   - released_in: (Movie → Year) [1:1]
   - is_sequel_to: (Movie → Movie) [1:1]

3. Creative:
   - directed_by: (Movie → Person) [1:N]
   - stars: (Movie → Person) [1:N]
   - plays_character: (Person → Character) [1:N]

4. Descriptive:
   - has_tag: (Movie → Tag) [1:N]
   - has_plot: (Movie → Plot) [1:N]

Format Rules:
1. Entity Formatting:
   - Proper capitalization
   - Full names for people
   - Years in YYYY format
   - Consistent naming

2. Relationship Formatting:
   - Use defined types only
   - Follow cardinality rules
   - Include confidence scores
   - Split multiple values

3. Validation Rules:
   - Check entity existence
   - Verify relationship validity
   - Validate temporal consistency
   - Score confidence

Example Input:
"The Matrix (1999) is a science fiction film directed by the Wachowskis, starring Keanu Reeves as Neo and Laurence Fishburne as Morpheus."

Example Output:
(The Matrix, is_a, film, 1.0)
(The Matrix, has_genre, Science Fiction, 1.0)
(The Matrix, released_in, 1999, 1.0)
(The Matrix, directed_by, Wachowskis, 1.0)
(The Matrix, stars, Keanu Reeves, 1.0)
(The Matrix, stars, Laurence Fishburne, 1.0)
(Keanu Reeves, plays_character, Neo, 1.0)
(Laurence Fishburne, plays_character, Morpheus, 1.0)

Now extract knowledge triples from this text, following the strict format:
{text}

Triples:"""
        )

        self.cot_prompt = PromptTemplate(
            input_variables=["text"],
            template="""You are a knowledge graph extraction expert. Think step by step to extract entities and their relationships from the following text.

Step 1: Entity Identification
- List all entities mentioned in the text
- Identify entity types (Movie, Person, Genre, Year, Tag)
- Resolve entity references and pronouns
- Track entity consistency

Step 2: Entity Classification
- Categorize each entity by type
- Verify entity existence
- Check entity relationships
- Score entity confidence

Step 3: Relationship Analysis
- Identify potential relationships
- Verify relationship validity
- Check temporal consistency
- Score relationship confidence

Step 4: Triple Formation
- Create (subject, predicate, object) triples
- Use only allowed relation types:
  * is_a: For categorization
  * has_genre: For movie genres
  * released_in: For release years
  * directed_by: For directors
  * stars: For main actors
  * plays_character: For character roles
  * has_tag: For user tags
  * is_sequel_to: For movie sequels
  * is_remake_of: For movie remakes
  * is_adaptation_of: For book adaptations
  * has_plot: For plot descriptions

Step 5: Validation
- Check triple consistency
- Verify entity existence
- Validate relationships
- Score triple confidence

Step 6: Output
- Format triples as (subject, predicate, object, confidence)
- Include only validated triples
- Maintain entity consistency
- Follow relationship rules

Text: {text}

Step-by-step extraction:"""
        )

        self.schema_guided_prompt = PromptTemplate(
            input_variables=["text"],
            template="""You are a knowledge graph extraction expert. Use the following schema to extract triples from the text.

Entity Type Definitions:
1. Movie:
   - Properties: title, year, genre
   - Constraints: unique title+year
   - Relationships: has_genre, released_in, directed_by, stars

2. Person:
   - Properties: name, role
   - Constraints: unique name
   - Relationships: directed_by, stars, plays_character

3. Genre:
   - Properties: name, type
   - Constraints: unique name
   - Relationships: is_a, has_genre

4. Year:
   - Properties: value
   - Constraints: YYYY format
   - Relationships: released_in

5. Tag:
   - Properties: name
   - Constraints: unique name
   - Relationships: has_tag

Relationship Type Definitions:
1. Categorization:
   - is_a: (Entity → Type) [1:1]
   - has_genre: (Movie → Genre) [1:N]

2. Temporal:
   - released_in: (Movie → Year) [1:1]
   - is_sequel_to: (Movie → Movie) [1:1]

3. Creative:
   - directed_by: (Movie → Person) [1:N]
   - stars: (Movie → Person) [1:N]
   - plays_character: (Person → Character) [1:N]

4. Descriptive:
   - has_tag: (Movie → Tag) [1:N]
   - has_plot: (Movie → Plot) [1:N]

Validation Rules:
1. Entity Validation:
   - Check required properties
   - Verify constraints
   - Validate relationships
   - Score confidence

2. Relationship Validation:
   - Check cardinality
   - Verify types
   - Validate constraints
   - Score confidence

Output Format:
{{
    "subject": "entity name",
    "subject_type": "entity type",
    "predicate": "relationship type",
    "object": "entity name",
    "object_type": "entity type",
    "confidence": 0.0-1.0
}}

Text: {text}

Triples:"""
        )

        self.self_consistency_prompt = PromptTemplate(
            input_variables=["text"],
            template="""You are a knowledge graph extraction expert. Extract and verify knowledge triples from the text using a self-consistency approach.

Step 1: Initial Extraction
- Extract all possible triples
- Include implicit relationships
- Consider context
- Track confidence

Step 2: Triple Verification
- Check explicit support in text
- Verify entity existence
- Validate relationships
- Score confidence:
  * 1.0: Explicitly stated
  * 0.8: Strongly implied
  * 0.6: Moderately implied
  * 0.4: Weakly implied
  * 0.2: Speculative

Step 3: Cross-Validation
- Check triple consistency
- Verify entity references
- Validate relationships
- Resolve conflicts

Step 4: Final Selection
- Keep only verified triples
- Include confidence scores
- Maintain consistency
- Follow format rules

Allowed Relationship Types:
- is_a: For categorization
- has_genre: For movie genres
- released_in: For release years
- directed_by: For directors
- stars: For main actors
- plays_character: For character roles
- has_tag: For user tags
- is_sequel_to: For movie sequels
- is_remake_of: For movie remakes
- is_adaptation_of: For book adaptations
- has_plot: For plot descriptions

Text: {text}

Step 1: Extract candidate triples.
Step 2: For each candidate triple, check if the text explicitly supports it.
Step 3: Output only the supported triples in (subject, predicate, object, confidence) format.

Final triples:"""
        )
        
        # Select prompt based on type
        prompt_map = {
            "basic": self.basic_prompt,
            "complex": self.complex_prompt,
            "conversation": self.conversation_prompt,
            "structured": self.structured_prompt,
            "cot": self.cot_prompt,
            "schema_guided": self.schema_guided_prompt,
            "self_consistency": self.self_consistency_prompt
        }
        return prompt_map.get(prompt_type, self.basic_prompt)
        
    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()
        
    def _parse_triples(self, response: str) -> List[Tuple[str, str, str, float]]:
        """Parse triples from LLM response using regex."""
        import re
        triples = []
        triple_pattern = re.compile(r'^\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,]+?)(?:\s*,\s*([0-9.]+))?\s*\)$')
        
        for line in response.strip().split('\n'):
            line = line.strip()
            match = triple_pattern.match(line)
            if match:
                subject = match.group(1).strip()
                predicate = match.group(2).strip()
                obj = match.group(3).strip()
                conf = float(match.group(4)) if match.group(4) else 1.0
                triples.append((subject, predicate, obj, conf))
        return triples
        
    def _vote_on_triples(self, all_triples: List[List[Tuple[str, str, str, float]]]) -> List[Tuple[str, str, str, float]]:
        """Apply self-consistency voting to select the most reliable triples."""
        # Count occurrences of each triple
        triple_counts = defaultdict(int)
        triple_confidences = defaultdict(list)
        
        for iteration_triples in all_triples:
            for triple in iteration_triples:
                # Use (subject, predicate, object) as key, ignoring confidence
                key = (triple[0], triple[1], triple[2])
                triple_counts[key] += 1
                triple_confidences[key].append(triple[3])
        
        # Select triples that meet the voting threshold
        voted_triples = []
        for (subject, predicate, obj), count in triple_counts.items():
            if count / self.num_iterations >= self.voting_threshold:
                # Average confidence across iterations
                avg_confidence = sum(triple_confidences[(subject, predicate, obj)]) / len(triple_confidences[(subject, predicate, obj)])
                voted_triples.append((subject, predicate, obj, avg_confidence))
        
        return voted_triples
        
    def extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract knowledge triples using iterative self-consistency voting."""
        # Prepare prompt text (for logging input tokens)
        if self.model_name == "granite":
            prompt_text = "<|start_of_role|>control<|end_of_role|>thinking<|end_of_text|>\n" + self.prompt.format(text=text)
        else:
            prompt_text = self.prompt.format(text=text)
        input_tokens = len(prompt_text.split())

        # Multiple extraction iterations
        all_triples = []
        total_latency = 0

        for _ in range(self.num_iterations):
            # Add slight variation to prompt for diversity
            varied_text = text + f" [Iteration {_+1}/{self.num_iterations}]"
            if self.model_name == "granite":
                varied_prompt_text = "<|start_of_role|>control<|end_of_role|>thinking<|end_of_text|>\n" + self.prompt.format(text=varied_text)
                # LLM call with timing
                start_time = time.time()
                response = self.chain.run(text=varied_prompt_text)
                iteration_latency = time.time() - start_time
                total_latency += iteration_latency
                # Log raw LLM response (LLM thoughts)
                raw_log_path = f"advance_{self.model_name}_{self.prompt_type}_raw.csv"
                raw_log_row = {
                    'timestamp': datetime.now().isoformat(),
                    'model_name': self.model_name,
                    'prompt_type': self.prompt_type,
                    'iteration': _ + 1,
                    'input_text': varied_text,
                    'raw_response': response
                }
                file_exists = os.path.isfile(raw_log_path)
                with open(raw_log_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=raw_log_row.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(raw_log_row)
                # Parse only <response>...</response> section if present
                import re
                match = re.search(r'<response>(.*?)</response>', response, re.DOTALL)
                triples_to_parse = match.group(1) if match else response
                iteration_triples = self._parse_triples(triples_to_parse)
            else:
                # LLM call with timing
                start_time = time.time()
                response = self.chain.run(text=varied_text)
                iteration_latency = time.time() - start_time
                total_latency += iteration_latency
                # Log raw LLM response (LLM thoughts)
                raw_log_path = f"advance_{self.model_name}_{self.prompt_type}_raw.csv"
                raw_log_row = {
                    'timestamp': datetime.now().isoformat(),
                    'model_name': self.model_name,
                    'prompt_type': self.prompt_type,
                    'iteration': _ + 1,
                    'input_text': varied_text,
                    'raw_response': response
                }
                file_exists = os.path.isfile(raw_log_path)
                with open(raw_log_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=raw_log_row.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(raw_log_row)
                # Parse triples from this iteration
                iteration_triples = self._parse_triples(response)
            all_triples.append(iteration_triples)
            # Print raw LLM response for debugging
            print(f"Iteration {_+1} raw response:", response)

        # Apply self-consistency voting
        voted_triples = self._vote_on_triples(all_triples)

        # Compute confidence stats
        confidences = [t[3] for t in voted_triples]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        min_conf = min(confidences) if confidences else 0
        max_conf = max(confidences) if confidences else 0

        # Logging: log to a separate file per model and prompt
        log_path = f"advance_{self.model_name}_{self.prompt_type}.csv"
        log_row = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'prompt_type': self.prompt_type,
            'input_tokens': input_tokens,
            'output_tokens': len(response.split()),
            'latency': total_latency / self.num_iterations,  # Average latency
            'num_triples': len(voted_triples),
            'avg_confidence': avg_conf,
            'min_confidence': min_conf,
            'max_confidence': max_conf,
            'triples': str(voted_triples),
            'text_sample': text[:100].replace('\n', ' '),
            'iterations': self.num_iterations,
            'voting_threshold': self.voting_threshold
        }

        file_exists = os.path.isfile(log_path)
        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=log_row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_row)

        # Add voted triples to Neo4j
        for t in voted_triples:
            self.add_triple(t[0], t[1], t[2], t[3])

        # Return triples in old format for compatibility
        return [(t[0], t[1], t[2]) for t in voted_triples]
    
    def add_triple(self, subject: str, predicate: str, obj: str, confidence: float = 1.0):
        """Add a triple to the knowledge graph."""
        # Sanitize predicate for Cypher
        safe_predicate = (
            predicate.strip()
            .replace('"', '')
            .replace("'", '')
            .replace(" ", "_")
            .replace("-", "_")
            .replace(".", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(":", "")
            .replace("&", "AND")
            .replace("+", "PLUS")
            .replace("*", "STAR")
            .replace("?", "QUESTION")
            .replace("!", "EXCLAMATION")
            .replace("@", "AT")
            .replace("#", "HASH")
            .replace("$", "DOLLAR")
            .replace("%", "PERCENT")
            .replace("^", "CARET")
            .replace("~", "TILDE")
            .replace("`", "BACKTICK")
            .replace("|", "PIPE")
            .replace("\\", "BACKSLASH")
            .replace(";", "SEMICOLON")
            .replace(",", "COMMA")
            .replace("<", "LT")
            .replace(">", "GT")
            .replace("=", "EQ")
            .upper()
        )
        
        with self.driver.session() as session:
            query = f"""
            MERGE (s:Entity {{name: $subject}})
            MERGE (o:Entity {{name: $object}})
            MERGE (s)-[r:{safe_predicate} {{confidence: $confidence, kg_name: $kg_name}}]->(o)
            """
            session.run(query, {
                "subject": subject,
                "object": obj,
                "confidence": confidence,
                "kg_name": self.kg_name
            })
            
    def query_triples(self, subject: str = None, predicate: str = None, obj: str = None) -> List[Tuple[str, str, str, float]]:
        """Query triples from the knowledge graph."""
        with self.driver.session() as session:
            # Build the query based on provided filters
            query_parts = ["r.kg_name = $kg_name"]
            params = {"kg_name": self.kg_name}
            
            if subject is not None:
                query_parts.append("s.name = $subject")
                params["subject"] = subject
            if predicate is not None:
                query_parts.append("type(r) = $predicate")
                params["predicate"] = predicate
            if obj is not None:
                query_parts.append("o.name = $object")
                params["object"] = obj
                
            where_clause = " AND ".join(query_parts)
            
            query = f"""
            MATCH (s:Entity)-[r]->(o:Entity)
            WHERE {where_clause}
            RETURN s.name as subject, type(r) as predicate, o.name as object, r.confidence as confidence
            """
            
            result = session.run(query, params)
            return [(record["subject"], record["predicate"], record["object"], record["confidence"]) 
                   for record in result] 
