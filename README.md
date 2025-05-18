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
```
.
├── models/                      # Ontologies (e.g., TAMOntology.ttl)
├── tests/                      
│   ├── data/                   # QA datasets, user profiles, and results
│   └── test.ipynb              # Notebook for running evaluation tests
├── src/                        
│   ├── KG_construction.py      # Knowledge Graph construction pipeline
│   ├── GraphRAG.py             # GraphRAG-based QA logic
│   ├── RAGAS_test.py           # Evaluation script using RAGAS
|   ├── app.py                  # Streamlit interface
│   ├── utility_function.py     # Preprocessing and helpers
│   └── ontology_parser.py      # Translation of ontology
├── requirements.txt            # Minimal dependencies
└── README.md
```

## Prerequisites

Before running the system, make sure you have:

1. **A Neo4j database instance** (local or via AuraDB)
2. **An OpenAI API key**
3. A `.env` file in the root directory with the following content:

```env
NEO4J_URI=neo4j+s://<your-neo4j-uri>
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=your_openai_key
ONTOLOGY_FILE=./models/TAMOntology.ttl
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit app

```bash
streamlit run app.py
```

### 3. Run tests from the notebook

Open the notebook:

```
tests/test.ipynb
```

Follow the sections to:

- Load synthetic user profiles and QA datasets
- Populate the Knowledge Graph
- Run and evaluate GraphRAG and standard RAG
- Visualize and export results

---

## Evaluation (via RAGAS)

The system is evaluated using the [RAGAS](https://github.com/explodinggradients/ragas) framework and the following metrics:

- **Context Precision**
- **Context Recall**
- **Answer Relevancy**
- **Faithfulness**
- **Semantic Similarity**

Each profile is evaluated individually, and results are stored as:

```
tests/data/results/results_graphRAG.csv
tests/data/results/results_RAG.csv
```

---

## Technologies Used

- Python 3.10+
- [Neo4j](https://neo4j.com/)
- [LangChain](https://www.langchain.com/)
- [RAGAS](https://github.com/explodinggradients/ragas)
- [OpenAI API](https://platform.openai.com/)
- [Streamlit](https://streamlit.io/)
- [OWLReady2](https://owlready2.readthedocs.io/)
- [RDFLib](https://rdflib.readthedocs.io/)

---

## Author

**Alexandra Tarantino**  
Master's Degree in Computer Engineering  
University of Salerno – DIEM  
Academic Year 2024–2025




