from neo4j import GraphDatabase
import os
import ontology_parser as parser
from dotenv import load_dotenv
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
    OnError,
)
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.schema import (
    SchemaBuilder,
)
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.components.resolver import (
    SinglePropertyExactMatchResolver,
)
from neo4j_graphrag.experimental.pipeline import Pipeline



load_dotenv()

# Neo4j connection setup
URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
ONTOLOGY_FILE = os.getenv("ONTOLOGY_FILE")

llm = OpenAILLM(
        model_name="gpt-4o",
        model_params={
            "response_format": {"type": "json_object"},
        },
    )
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")



async def add_data_to_graph(user_input):
    """Processes user input and adds an activity to the Knowledge Graph using GraphRAG."""
    
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    
    pipe = Pipeline()
    
    #Components of the pipeline
    pipe.add_component(
        FixedSizeSplitter(chunk_size=4000, chunk_overlap=200),
        "splitter",
    )
    pipe.add_component(TextChunkEmbedder(embedder=embeddings), "chunk_embedder")
    pipe.add_component(SchemaBuilder(), "schema")
    pipe.add_component(
        LLMEntityRelationExtractor(
            llm=llm,
            on_error=OnError.IGNORE,
        ),
        "extractor",
    )
    pipe.add_component(Neo4jWriter(driver = driver), "writer")
    
    #Pipeline definition 
    pipe.connect("splitter", "chunk_embedder", input_config={"text_chunks": "splitter"})
    pipe.connect("schema", "extractor", input_config={"schema": "schema"})
    pipe.connect(
        "chunk_embedder", "extractor", input_config={"chunks": "chunk_embedder"}
    )
    pipe.connect(
        "extractor",
        "writer",
        input_config={"graph": "extractor"},
    )
    
    
    #sostiutisci in user_input tutti gli a capo con lo spazioe
    input = user_input.replace("\n", " ")
    
    #Pipeline input
    pipe_inputs = {
        "splitter": {
            "text": input,
        },
        "schema":  parser.parse_ontology(ONTOLOGY_FILE),
    }
    
    
    #Execute the pipeline
    response = await pipe.run(pipe_inputs)
    driver.close()
    
    driver.close()
    return response.result
    

    
async def resolve_entities():
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    
    resolver = SinglePropertyExactMatchResolver(driver)
    res = await resolver.run()
    
    driver.close()
    return res

    
    
if __name__ == "__main__":
    
    schema = parser.parse_ontology(ONTOLOGY_FILE)
    