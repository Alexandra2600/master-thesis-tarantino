import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
import ontology_parser as ontology_parser  
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
    OnError,
)
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.schema import SchemaBuilder
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.components.resolver import SinglePropertyExactMatchResolver
from neo4j_graphrag.experimental.pipeline import Pipeline

# Load environment variables
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
ONTOLOGY_FILE = os.getenv("ONTOLOGY_FILE")

# Setup the LLM and Embedding model
llm = OpenAILLM(
    model_name="gpt-4o",
    model_params={"response_format": {"type": "json_object"}},
)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")


async def add_user_input_to_kg(user_input: str):
    """
    Extracts structured knowledge from user input using a GraphRAG pipeline 
    and writes it to the Neo4j Knowledge Graph.
    
    Args:
        user_input: Raw natural language input provided by the user.

    Returns:
        Pipeline execution result containing extracted graph data.
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    pipeline = Pipeline()

    # Add pipeline components
    pipeline.add_component(FixedSizeSplitter(chunk_size=4000, chunk_overlap=200), "splitter")
    pipeline.add_component(TextChunkEmbedder(embedder=embedding_model), "embedder")
    pipeline.add_component(SchemaBuilder(), "schema")
    pipeline.add_component(
        LLMEntityRelationExtractor(llm=llm, on_error=OnError.IGNORE),
        "extractor"
    )
    pipeline.add_component(Neo4jWriter(driver=driver), "writer")

    # Define pipeline flow
    pipeline.connect("splitter", "embedder", input_config={"text_chunks": "splitter"})
    pipeline.connect("schema", "extractor", input_config={"schema": "schema"})
    pipeline.connect("embedder", "extractor", input_config={"chunks": "embedder"})
    pipeline.connect("extractor", "writer", input_config={"graph": "extractor"})

    # Prepare input data
    clean_input = user_input.replace("\n", " ")
    schema = ontology_parser.parse_ontology(ONTOLOGY_FILE)

    pipeline_inputs = {
        "splitter": {"text": clean_input},
        "schema": schema,
    }

    # Execute pipeline
    response = await pipeline.run(pipeline_inputs)
    driver.close()
    return response.result


async def resolve_kg_entities():
    """
    Resolves nodes in the Neo4j graph using exact match logic based on single properties.
    
    Returns:
        List of updated or merged entities.
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    resolver = SinglePropertyExactMatchResolver(driver)
    result = await resolver.run()
    driver.close()
    return result