from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import HybridCypherRetriever
from neo4j_graphrag.retrievers import HybridRetriever
from neo4j_graphrag.generation import RagTemplate
from neo4j_graphrag.generation import GraphRAG
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pprint import pprint
import json
import os
from dotenv import load_dotenv
import utility_function


load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")


driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
embedder = OpenAIEmbeddings(model="text-embedding-3-large")
# llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})
llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0})
llm_el = ChatOpenAI(model="gpt-4o-mini")

num_k = 3

cypher_query = """
    // 1) Expand 2-3 hops in the entity graph and retrieve relationships
    WITH node AS chunk
    MATCH (chunk)<-[:FROM_CHUNK]-(entity)-[relList*1]-(nb)
    WHERE NONE(r IN relList WHERE type(r) = "FROM_CHUNK")
    UNWIND relList AS rel

    // 2) Collect relationships, text chunks, and entities
    WITH collect(DISTINCT chunk) AS chunks, 
        collect(DISTINCT rel) AS rels, 
        collect(DISTINCT entity) AS entities, 
        collect(DISTINCT nb) AS neighbors

    // 3) Extract properties of entities only if they exist
    WITH chunks, rels, 
        [e IN entities + neighbors WHERE size(keys(e)) > 0 | 
            e.name + " (" + labels(e)[0] + ") → " + 
            apoc.convert.toJson(properties(e))  
        ] AS entity_info

    // 4) Format and return structured context
    RETURN 
    apoc.text.join([c IN chunks | c.text], ' -&- ') + " -&&- " +
    apoc.text.join(entity_info, ' -&- ') + " -&&- " +
    apoc.text.join([r IN rels | 
        startNode(r).name + ' - ' + type(r) + ' ' + ' -> ' + endNode(r).name], ' -&- ') 
    AS info;
    """    

hybridCypherRetr = HybridCypherRetriever(
        driver=driver,
        vector_index_name="textChuck",
        fulltext_index_name="textFulltext",
        retrieval_query=cypher_query,
        embedder=embedder,
    )
    
    
def ask_data(question): 

    #Only respond with information mentioned in the Context. Do not inject any speculative information not mentioned   
    rag_template = RagTemplate(template='''Answer the Question using the following Context. 
        # Question:
        {query_text}
        
        # Context:
        {context}

        # Answer:
        ''', expected_inputs=['query_text', 'context'])

    rag = GraphRAG(retriever=hybridCypherRetr, llm=llm, prompt_template=rag_template)
    response = rag.search(query_text=question, retriever_config={"top_k": num_k})
    return response.answer
    


def ask_data_RAG(question):
    retriever = HybridRetriever(
        driver=driver,
        vector_index_name="textChuck",
        fulltext_index_name="textFulltext",
        embedder=embedder,
    )

    rag = GraphRAG(retriever=retriever, llm=llm)
    response = rag.search(query_text=question, retriever_config={"top_k": num_k})
    
    return(response.answer)
      


def return_node(question):  
    
    retriever = HybridRetriever(
        driver=driver,
        vector_index_name="textChuck",
        fulltext_index_name="textFulltext",
        embedder=embedder,
    )
    
    response = retriever.search(query_text=question, top_k=num_k)
    nodes = []
    for i in response.items:
        content = i.content
        text = content.split("'text': '")[1].split("', '")[0]
        nodes.append(text)
    return nodes



def return_context(question):
    response = hybridCypherRetr.search(query_text=question, top_k=num_k)
    content = response.items[0].content
    clear = utility_function.clean_results(content)
    return clear


# def ask_llm(context, question):
      
#     from langchain.prompts import PromptTemplate

#     prompt = PromptTemplate(
#         input_variables=["context", "question"],
#         template="""
#             You are given a list of context sentences and a question. Your task is to:

#             1. Provide a clear and concise answer to the question using only the information from the list.
#             2. Select up to 3 items from the context list that justify your answer.

#             Return your output strictly in the following JSON format:

#             {{
#             "answer": "<your answer>",
#             "support": [
#                 "<sentence 1>",
#                 "<sentence 2>",
#                 "<sentence 3>"
#             ]
#             }}

#             If fewer than 3 sentences are needed, return only the relevant ones.

#             Context (as list of sentences):
#             {context}

#             Question:
#             {question}
#             """
#     )
    
#     formatted_prompt = prompt.format(context=context, question=question)
#     response = llm_el.predict(formatted_prompt)
    
#     return response.strip()

    


if __name__ == "__main__":
    question = "What activities are associated with my 'Scandinavian Pottery Workshop Series' project?"
    
    #print(return_context(question))
    
    #nodes = return_node(question)
    #print(nodes)
    
    #print(ask_data(question))
    
    #print(ask_data_RAG(question))