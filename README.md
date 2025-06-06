# LLM-based Knowledge Graph Extraction and Evaluation

This repository contains a robust, modular pipeline for extracting and evaluating knowledge graphs (KGs) from the MovieLens 1M dataset using multiple large language models (LLMs) and advanced evaluation strategies.

## Features

- Multi-model, multi-prompt KG extraction (Mistral, Llama2, Qwen, Gemma, Deepseek, Granite, etc.)
- Local LLM inference via Ollama
- Self-consistency voting and checkpointing
- Neo4j graph storage and querying
- Golden KG construction for deterministic benchmarking
- Advanced evaluation: precision, recall, F1, graph overlap, density, hallucination rate, path metrics, and more
- Dual logging for traceability

## Directory Structure

- `src/` — Core extraction and KG builder modules
- `scripts/` — Evaluation and utility scripts
- `data/` — Mapping files, (optionally) small sample data
- `KG_evaluation/` — Example evaluation outputs
- `docker-compose.yml` — For Neo4j and Ollama setup

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

- **Extract KG:**
  ```bash
  python src/advanced_multi_llm_example.py --models granite --prompts basic --dataset data/ml-1m --start_index 0
  ```
- **Evaluate KG:**
  ```bash
  python scripts/advanced_kg_evaluation.py --kg_name "granite_basic_kg" --dataset data/ml-1m
  ```

## Citing

If you use this codebase, please cite:
- Ji et al., "A Survey on Knowledge Graphs: Representation, Acquisition, and Applications", 2022.
- Paulheim, "Knowledge Graph Refinement: A Survey of Approaches and Evaluation Methods", 2017.

## License

MIT License 