from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
ONTOLOGY_FILE = os.getenv("ONTOLOGY_FILE")

# Initialize the LLM model for prompt-based operations
llm_el = ChatOpenAI(model="gpt-4o-mini")


def process_text(text: str, user_name: str, current_date: str) -> str:
    """
    Replace temporal expressions with actual dates and annotate first-person references with the user's name.

    Args:
        text: Natural language input.
        user_name: The user's name.
        current_date: Date of reference for temporal normalization.

    Returns:
        Processed and natural language text.
    """
    template = """
        You are an advanced AI assistant that processes user sentences by replacing temporal references 
        such as today, tomorrow, yesterday, next week, two days ago, last Monday, and other similar terms 
        with their actual date based on the given reference date: {current_date}.
        Additionally, every time there is a first-person reference, add the user's name: {user_name} next to it in parentheses, 
        for example, "I" becomes "I (Alex)".
        Ensure the final output is grammatically correct, fluent, and natural.

        Here is the sentence to process:
        "{text}"

        Provide only the corrected sentence as output without any additional explanations.
    """
    prompt = PromptTemplate(input_variables=["user_name", "text", "current_date"], template=template)
    formatted_prompt = prompt.format(user_name=user_name, text=text, current_date=current_date)
    response = llm_el.invoke(formatted_prompt)
    return response.content


def process_date(text: str, current_date: str) -> str:
    """
    Replace only date/time references with actual values based on a provided current date.

    Args:
        text: Natural language input.
        current_date: Date of reference for normalization.

    Returns:
        Updated sentence with normalized temporal expressions.
    """
    template = """
        You are an advanced AI assistant that processes user sentences by replacing temporal references
        such as today, tomorrow, and others with their actual date based on the given reference date: {current_date}.
        Ensure the final output is grammatically correct and natural.
        If there are no references to dates, return the original sentence.

        Here is the sentence to process:
        "{text}"

        Provide only the corrected sentence as output without any additional explanations.
    """
    prompt = PromptTemplate(input_variables=["current_date", "text"], template=template)
    formatted_prompt = prompt.format(current_date=current_date, text=text)
    response = llm_el.predict(formatted_prompt)
    return response.strip()


def add_indexes():
    """
    Create vector and fulltext indexes on the Neo4j database if they do not already exist.
    These indexes are used for semantic search and similarity-based retrieval.
    """
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    with driver.session() as session:
        session.run("""
            CREATE VECTOR INDEX textChuck IF NOT EXISTS
            FOR (c:Chunk)
            ON c.embedding
            OPTIONS {indexConfig: {
                `vector.dimensions`: 3072,
                `vector.similarity_function`: 'cosine'
            }}
        """)
        session.run("""
            CREATE FULLTEXT INDEX textFulltext IF NOT EXISTS
            FOR (c:Chunk)
            ON EACH [c.text]
        """)
    driver.close()


def reset_knowledge_graph():
    """
    Remove all nodes and relationships from the Neo4j graph database.
    Useful for development and testing environments.
    """
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    driver.close()


def extract_unique_chunks(text: str) -> list:
    """
    Process a raw string and return a de-duplicated list of extracted lines.

    Args:
        text: Text formatted using custom delimiters ('-&&-' for sections, '-&-' for entries).

    Returns:
        List of unique text lines.
    """
    sections = text.split('-&&-')
    if len(sections) < 2:
        return []

    all_lines = sections[0].split('-&-') + sections[1].split('-&-')

    def remove_duplicates(data_list):
        seen = set()
        return [item for item in data_list if not (item in seen or seen.add(item))]

    return remove_duplicates(all_lines)

