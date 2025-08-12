# LLM-based Knowledge Graph Extraction and Evaluation

This repository contains a robust, modular pipeline for extracting and evaluating knowledge graphs (KGs) from the MovieLens 1M dataset using multiple large language models (LLMs) and advanced evaluation strategies.

## Features

- Multi-model, multi-prompt KG extraction (Mistral, Llama2, Qwen, Gemma, Deepseek, Granite, etc.)
- Local LLM inference via Ollama
- Self-consistency voting and checkpointing
- Neo4j graph storage and querying
- Golden KG construction for deterministic benchmarking
- **Advanced evaluation with three comprehensive strategies:**
  - **Strategy 1:** Exact match evaluation against DBpedia ground truth
  - **Strategy 2:** Semantic/soft matching against Golden KG (sentence-BERT + SpaCy NER)
  - **Strategy 3:** Advanced graph-based metrics with fuzzy matching and hallucination detection
  - **Strategy 4 Enhanced:** Embedding-based Graph Metrics
  - **Strategy 5:** Enhanced Embedding-based Semantic Similarity
  - **Strategy 6:** Qwen Embedding-based Evaluation
- **Smart normalization and mapping:** Relation mapping tables, fuzzy matching, and entity normalization
- **Comprehensive metrics:** Precision, recall, F1, graph overlap, density, component ratio, path metrics, hallucination rate
- Dual logging for traceability

## Directory Structure

- `src/` — Core extraction and KG builder modules
- `scripts/` — Evaluation and utility scripts
  - `str1_evaluate.py` — Exact match evaluation against DBpedia
  - `str2_evaluate.py` — Semantic matching against Golden KG
  - `str3_evaluate.py` — Advanced graph-based metrics with fuzzy matching
  - `str3_enhanced_embedding_evaluate.py` - Embedding-based Graph Metrics
  - `str5_embedding_similarity_evaluate.py` - Embedding-based Graph Metrics
  - `str6_qwen_embedding_evaluate_fixed.py` - Qwen Embedding-based Evaluation
  - `advanced_kg_evaluation.py` — CLI dispatcher for evaluation strategies
  - `kg_evaluator_module.py` — Reusable evaluation module
- `KG_evaluation/` — Evaluation outputs and results
- `docker-compose.yml` — For Neo4j and Ollama setup

## Dataset

This project uses the [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/) and optionally the [MovieLens Latest](https://grouplens.org/datasets/movielens/latest/) datasets.

**Please download the dataset(s) manually from the official [GroupLens website](https://grouplens.org/datasets/movielens/), and place them in the `data/` directory as follows:**

- `data/ml-1m/`
- `data/ml-latest/`
- `data/ml-latest-small/`

> **Note:** The datasets are not included in this repository due to their size and licensing.

## Setup

1. **Clone the repo**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Start Neo4j and Ollama:**
   ```bash
   docker-compose up -d
   ```
4. **Download MovieLens data:**
   (You may need to manually download and place the MovieLens 1M dataset in the `data/` directory.)

## Usage

### KG Extraction

```bash
python src/advanced_multi_llm_example.py --models granite --prompts basic --dataset data/ml-1m --start_index 0
```

### KG Evaluation

The evaluation system supports three comprehensive strategies:

#### **Strategy 1: Exact Match Evaluation**
Compares LLM-extracted triples against DBpedia ground truth using exact matching.

```bash
python scripts/advanced_kg_evaluation.py --strategy str1 --kg_name "mistral_basic_kg"
```

**Metrics:** Precision, Recall, F1-score, Exact match rate

#### **Strategy 2: Semantic/Soft Matching**
Uses sentence-BERT embeddings and SpaCy NER for semantic similarity matching against a Golden KG.

```bash
python scripts/advanced_kg_evaluation.py --strategy str2 --kg_name "llama2_conversation_kg"
```

**Metrics:** Semantic precision, recall, F1, entity similarity, relation similarity

#### **Strategy 3: Advanced Graph-Based Metrics**
Comprehensive evaluation with fuzzy matching, graph structure analysis, and hallucination detection.

```bash
python scripts/advanced_kg_evaluation.py --strategy str3 --kg_name "qwen_cot_kg"
```

**Features:**
- **Smart Normalization:** Handles underscores, hyphens, punctuation, case differences
- **Relation Mapping:** Maps LLM relation phrases to DBpedia ontology relations
- **Fuzzy Matching:** Uses Levenshtein distance and substring matching
- **Graph Metrics:** Node/edge overlap, density, component ratio, path overlap
- **Hallucination Detection:** Identifies triples not present in ground truth
- **Detailed Logging:** Tracks fuzzy match types and examples

**Metrics:**
- Node overlap, Edge overlap, Graph density, Component ratio
- Path overlap (shortest paths), Hallucination rate
- Fuzzy match statistics (exact, fuzzy, postfix/prefix, underscore/hyphen differences)

### Strategy 4 Enhanced: Embedding-based Graph Metrics
Advanced evaluation using sentence-transformer embeddings for semantic similarity, with flexible ground truth selection.

```bash
# With DBpedia ground truth (default)
python scripts/str3_enhanced_embedding_evaluate.py --kg_name "mistral_basic_kg"

# With Golden KG ground truth
python scripts/str3_enhanced_embedding_evaluate.py --kg_name "mistral_basic_kg" --ground_truth golden_kg

# With custom embedding model and threshold
python scripts/str3_enhanced_embedding_evaluate.py --kg_name "mistral_basic_kg" --ground_truth golden_kg --embedding_model "all-mpnet-base-v2" --similarity_threshold 0.8
```

**Features:**
- Embedding-based semantic matching (replaces fuzzy string matching)
- Flexible ground truth selection (DBpedia or Golden KG)
- Configurable embedding models and similarity thresholds
- Comprehensive graph metrics with semantic understanding
- Enhanced hallucination detection

### Strategy 5: Enhanced Embedding-based Semantic Similarity
Uses sentence-transformers with cosine similarity for nuanced semantic matching and similarity scoring.

```bash
python scripts/str5_embedding_similarity_evaluate.py --kg_name "mistral_basic_kg" --sample_size 1000
```

**Features:**
- Cosine similarity (provides similarity scores 0-100% for each triple)
- Detailed analysis (shows top matches and similarity distributions)
- Configurable thresholds (adjustable similarity thresholds for quality assessment)
- Multiple models (support for different sentence-transformer models)
- Similarity distribution (categorizes triples by similarity ranges)

### Strategy 6: Qwen Embedding-based Evaluation
Uses Qwen3-Embedding models for comparison with sentence-transformer approaches.

```bash
python scripts/str6_qwen_embedding_evaluate_fixed.py --kg_name "mistral_basic_kg" --sample_size 500
```

**Features:**
- Qwen embeddings (uses Qwen3-Embedding model for embedding generation)
- Comparison analysis (direct comparison with sentence-transformer results)
- Model-specific insights (understanding of Qwen's semantic understanding)
- Performance analysis (evaluation of Qwen embeddings vs. established models)


```

## Evaluation Output

All evaluation results are saved in the `KG_evaluation/` directory with consistent naming:
```
evaluate_<modelname>_<prompt>_<strategy>.csv
```

Example: `evaluate_mistral_basic_str3.csv`

### Output Format

Each CSV contains:
- **Basic metrics:** Precision, recall, F1-score
- **Graph metrics:** Node/edge overlap, density, component ratio, path overlap
- **Advanced metrics:** Hallucination rate, embedding similarity statistics
- **Debug information:** Sample matches, normalization examples

## Configuration

### CLI Arguments
- `--kg_name`: Name of the LLM-generated KG in Neo4j
- `--ground_truth`: Ground truth source (`dbpedia` or `golden_kg`)
- `--embedding_model`: Sentence-transformer model (default: `all-MiniLM-L6-v2`)
- `--similarity_threshold`: Similarity threshold for matching (default: 0.7)
- `--sample_size`: Number of triples to evaluate (for large KGs)

### Default Models
- **Embedding Model:** `all-MiniLM-L6-v2`
- **Judge Model:** `llama2:latest` (for Strategy 4)
- **Qwen Model:** `Qwen/Qwen3-Embedding-0.6B` (for Strategy 6)

## Key Metrics

### Graph Structure Metrics
- **Node Overlap:** Intersection of nodes between extracted and reference KGs
- **Edge Overlap:** Intersection of edges between extracted and reference KGs
- **Density Ratio:** Comparison of graph connectivity
- **Component Ratio:** Assessment of graph fragmentation
- **Path Overlap:** Multi-hop path similarity

### Quality Metrics
- **Hallucination Rate:** Proportion of unsupported triples
- **Semantic Similarity:** Embedding-based similarity scores
- **Match Distribution:** Classification of matches by similarity level
- **Coverage Percentage:** Proportion of reference knowledge captured

