from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from neo4j import GraphDatabase
from pprint import pprint
import os

from dotenv import load_dotenv

load_dotenv()
URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
ONTOLOGY_FILE = os.getenv("ONTOLOGY_FILE")

llm_el = ChatOpenAI(model="gpt-4o-mini")


def process_text(text, user_name, current_date):
    
    template = """
        You are an advanced AI assistant that processes user sentences by replacing temporal references 
        such as today, tomorrow, yesterday, next week, two days ago, last Monday, and other similar terms 
        with their actual date based on the given reference date: {current_date}.
        Additionally, every time is a first-person references add the user's name: {user_name} next to it in parentheses, 
        for example, "I" becomes "I (Alex)".
        
        Ensure the final output is grammatically correct, fluent, and natural.

        Here is the sentence to process:
        "{text}"

        Provide only the corrected sentence as output without any additional explanations.
    """

    
    prompt = PromptTemplate(
        input_variables=["user_name", "text"],
        template=template
    )
    
    formatted_prompt = prompt.format(user_name=user_name, text=text, current_date=current_date)
    response = llm_el.invoke(formatted_prompt)
    
    return response.content


def process_date(text, current_date):

        
    template = """
    You are an advanced AI assistant that processes user sentences by replacing temporal references
    such as today, tomorrow, and other with their actual date based 
    on the given reference date: {current_date}. Ensure the final output is grammatically correct and natural.
    If there are no references to dates return the original sentence.
    
    Here is the sentence to process:
    "{text}"
    
    Provide only the corrected sentence as output without any additional explanations.
    """
    
    prompt = PromptTemplate(
        input_variables=["current_date", "user_name", "text"],
        template=template
    )
    
    formatted_prompt = prompt.format(current_date=current_date, text=text)
    response = llm_el.predict(formatted_prompt)
    
    return response.strip()


### Da rivedere
def add_indexes():
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    with driver.session() as session:
        result = session.run("""
                                CREATE VECTOR INDEX textChuck IF NOT EXISTS
                                FOR (c:Chunk)
                                ON c.embedding
                                OPTIONS {indexConfig: {
                                `vector.dimensions`: 3072,
                                `vector.similarity_function`: 'cosine' } }
                             """)
        result.append(session.run("""
                                CREATE FULLTEXT INDEX nameFullText IF NOT EXISTS
                                FOR (n:Entity)
                                ON (n.name)
                             """))
        
        return(result)
        
        
def clean_results(text):

    # Dividere in 3 parti principali usando '\n\n'
    sections = text.split('-&&-')
    
    all_lines = sections[0].split('-&-') + sections[1].split('-&-') 

    # Funzione per rimuovere duplicati
    def remove_duplicates(data_list):
        seen = set()
        result = []
        for item in data_list:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    # Rimuovere elementi duplicati dalla lista unificata
    unique_lines = remove_duplicates(all_lines)

    # Stampare la lista risultante
    return unique_lines



if __name__ == "__main__":
    user_input = "Today I have to work on the computer and tomorrow I have a meeting with John"
    user_name = "Alex"
    current_date = "2025/03/11"
    processed_text = process_text(user_input, user_name, current_date)
    print(processed_text)
