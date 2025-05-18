import os
from dotenv import load_dotenv
from pprint import pprint
from neo4j import GraphDatabase

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG, RagTemplate
from neo4j_graphrag.retrievers import HybridCypherRetriever, HybridRetriever

import utils

# Load environment variables
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Initialize LLM and embedder
embedder_model = OpenAIEmbeddings(model="text-embedding-3-large")
llm_model = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0})
chat_llm = ChatOpenAI(model="gpt-4o-mini")  # Optional LangChain LLM

# Number of retrieved items
TOP_K = 3

# Custom Cypher query to extract 2-3 hop context from the entity graph
CONTEXT_CYPHER_QUERY = """
    WITH node AS chunk
    MATCH (chunk)<-[:FROM_CHUNK]-(entity)-[relList*1]-(nb)
    WHERE NONE(r IN relList WHERE type(r) = "FROM_CHUNK")
    UNWIND relList AS rel
    WITH collect(DISTINCT chunk) AS chunks, 
         collect(DISTINCT rel) AS rels, 
         collect(DISTINCT entity) AS entities, 
         collect(DISTINCT nb) AS neighbors
    WITH chunks, rels, 
         [e IN entities + neighbors WHERE size(keys(e)) > 0 | 
            e.name + " (" + labels(e)[0] + ") → " + apoc.convert.toJson(properties(e))] AS entity_info
    RETURN 
        apoc.text.join([c IN chunks | c.text], ' -&- ') + " -&&- " +
        apoc.text.join(entity_info, ' -&- ') + " -&&- " +
        apoc.text.join([r IN rels | 
            startNode(r).name + ' - ' + type(r) + ' -> ' + endNode(r).name], ' -&- ') 
    AS info;
"""


def answer_graphRAG(question: str) -> str:
    """
    Answer a question using HybridCypherRetriever and a custom Cypher-based context query.

    Args:
        question: The user's question in natural language.

    Returns:
        Generated answer using GraphRAG.
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    retriever = HybridCypherRetriever(
        driver=driver,
        vector_index_name="textChuck",
        fulltext_index_name="textFulltext",
        retrieval_query=CONTEXT_CYPHER_QUERY,
        embedder=embedder_model,
    )

    prompt_template = RagTemplate(
        template="""
        Answer the Question using the following Context. 
        # Question:
        {query_text}

        # Context:
        {context}

        # Answer:
        """,
        expected_inputs=["query_text", "context"]
    )

    rag = GraphRAG(retriever=retriever, llm=llm_model, prompt_template=prompt_template)
    response = rag.search(query_text=question, retriever_config={"top_k": TOP_K})
    driver.close()
    return response.answer


def answer_RAG(question: str) -> str:
    """
    Answer a question using standard HybridRetriever without Cypher query.

    Args:
        question: User question.

    Returns:
        Answer string.
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    retriever = HybridRetriever(
        driver=driver,
        vector_index_name="textChuck",
        fulltext_index_name="textFulltext",
        embedder=embedder_model,
    )

    rag = GraphRAG(retriever=retriever, llm=llm_model)
    response = rag.search(query_text=question, retriever_config={"top_k": TOP_K})
    driver.close()
    return response.answer


def get_graphRAG_context(question: str) -> list:
    """
    Retrieve the structured knowledge graph context used by GraphRAG (nodes, properties, relationships).

    This function uses a HybridCypherRetriever with a custom Cypher query to retrieve
    a multi-hop subgraph relevant to the question. The raw result is then cleaned and deduplicated.

    Args:
        question: The user query in natural language.

    Returns:
        A list of structured context elements extracted from the knowledge graph.
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    retriever = HybridCypherRetriever(
        driver=driver,
        vector_index_name="textChuck",
        fulltext_index_name="textFulltext",
        retrieval_query=CONTEXT_CYPHER_QUERY,
        embedder=embedder_model,
    )

    response = retriever.search(query_text=question, top_k=TOP_K)
    driver.close()

    raw_context = response.items[0].content
    return utils.extract_unique_chunks(raw_context)


def get_RAG_context(question: str) -> list:
    """
    Retrieve the original user input texts stored in the graph and used as context in RAG retrieval.

    This function uses a HybridRetriever to search relevant chunks from the graph database.
    It returns only the raw text that was originally inserted by the user.

    Args:
        question: A natural language query from the user.

    Returns:
        A list of raw user input texts retrieved as RAG context.
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    retriever = HybridRetriever(
        driver=driver,
        vector_index_name="textChuck",
        fulltext_index_name="textFulltext",
        embedder=embedder_model,
    )

    response = retriever.search(query_text=question, top_k=TOP_K)
    driver.close()

    text_inputs = []
    for item in response.items:
        content = item.content
        try:
            text = content.split("'text': '")[1].split("', '")[0]
            text_inputs.append(text)
        except IndexError:
            continue  

    return text_inputs