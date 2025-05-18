# Knowledge Graphs for Time Assistive Management in Cognitive Impairment
This repository contains the full implementation of a master’s thesis project focused on the use of **Knowledge Graphs (KGs)** to support **time and daily activity management** in individuals affected by **Mild Cognitive Impairment (MCI)** . The system is designed to help users track past actions, plan future ones, and interact naturally through language using a semantically structured knowledge base.

## Project Overview

The system integrates:

- A **natural language interface** to collect user input (e.g., diary-style interactions)
- A **semantic layer** based on a custom OWL ontology (TAMOntology)
- A **Neo4j Knowledge Graph** for storing and querying structured user data
- A **GraphRAG pipeline** (graph-based retrieval-augmented generation) for answering user questions
- **Evaluation metrics** from the RAGAS framework to assess precision, recall, and relevance

## Repository Structure
.
├── models/                      # Ontologies (e.g., TAMOntology.ttl)
├── tests/
│   ├── data/                   # QA datasets, user profiles, and results
│   └── test.ipynb              # Notebook for running evaluation tests
├── src/
│   ├── KG_construction.py      # Knowledge Graph construction pipeline
│   ├── GraphRAG.py             # GraphRAG-based QA logic
│   ├── RAGAS_test.py           # Evaluation script using RAGAS
│   ├── utility_function.py     # Preprocessing and helpers
|   ├── app.py                  # Streamlit interface
│   └── ontology_parser.py      # Translation of ontolgy               
├── requirements.txt            # Minimal dependencies
└── README.md
