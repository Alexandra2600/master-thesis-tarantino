from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import HybridRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from neo4j import GraphDatabase

import os
from dotenv import load_dotenv
load_dotenv()

# tag::setup[]
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(uri, auth=(username, password))
# end::setup[]

# tag::embedder[]
embedder = OpenAIEmbeddings(model="text-embedding-3-large")
# end::embedder[]

# tag::retriever[]
retriever = HybridRetriever(
    driver=driver,
    vector_index_name="textChuck",
    fulltext_index_name="textFulltext",
    embedder=embedder,
    return_properties=["text"],
)
# end::retriever[]

# tag::graphrag[]
from neo4j_graphrag.generation import GraphRAG

llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})
rag = GraphRAG(retriever=retriever, llm=llm)


# Esecuzione della conversazione in loop
while True:
    q = input("> ")
    
    response = rag.search(query_text=q, retriever_config={"top_k": 5})
    print(response.answer)
    
# end::graphrag[]

