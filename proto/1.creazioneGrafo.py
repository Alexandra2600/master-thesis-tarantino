import asyncio
import logging.config
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm.openai_llm import OpenAILLM

from rdflib import Graph

load_dotenv()

# Set log level to DEBUG for all neo4j_graphrag.* loggers
logging.config.dictConfig(
    {
        "version": 1,
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
            }
        },
        "loggers": {
            "root": {
                "handlers": ["console"],
            },
            "neo4j_graphrag": {
                "level": "DEBUG",
            },
        },
    }
)

# Connect to the Neo4j database
URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
driver = GraphDatabase.driver(URI, auth=AUTH)


#text_splitter = FixedSizeSplitter(chunk_size=150, chunk_overlap=20) non splitto perchè divide parole a metà e non è utile 


openaiKey = os.getenv("OPENAI_API_KEY")
print(openaiKey)
embedder = OpenAIEmbeddings(model="text-embedding-3-large")

from schemaFromOnto import getSchemaFromOnto
g = Graph()
neo4j_schema = getSchemaFromOnto("ontos/testOnt.ttl")   

print(neo4j_schema.entities)



llm = OpenAILLM(
    model_name="gpt-4o",
    model_params={
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
        "seed": 123
    },
)

pipeline = SimpleKGPipeline(
    driver=driver,
    #text_splitter=text_splitter, non utilizzato
    embedder=embedder,
    entities=neo4j_schema.entities,
    relations=neo4j_schema.relations,
    potential_schema=neo4j_schema.potential_schema,
    llm=llm,
    on_error="IGNORE",
    from_pdf=False,
)

#apri tutti i file presenti in content
with open('content/event_story_5.txt', 'r') as file:
   content = file.read().replace('\n', '')

asyncio.run(
    pipeline.run_async(
        text= content
    )
)

driver.close()