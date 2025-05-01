from neo4j import GraphDatabase
import os
import graphviz
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
    SchemaEntity,
    SchemaProperty,
    SchemaRelation,
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

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
llm = OpenAILLM(
        model_name="gpt-4o",
        model_params={
            "response_format": {"type": "json_object"},
        },
    )
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")



async def add_data_to_graph(user_input):
    """Processes user input and adds an activity to the Knowledge Graph using GraphRAG."""
    
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
    
    schema1 = {
            "entities": [
                SchemaEntity(
                    label="Event",
                    properties=[
                        SchemaProperty(name="name", type="STRING" ),
                        SchemaProperty(name="description", type="STRING"),
                        SchemaProperty(name="onDate", type="DATE"),
                        SchemaProperty(name="startTime", type="LOCAL_DATETIME"),
                        SchemaProperty(name="endTime", type="LOCAL_DATETIME"),
                        SchemaProperty(name="statusEvent", type="STRING", description="The status of the event which can take only the following values: “scheduled” or “completed"),
                        SchemaProperty(name="priority", type="STRING", description="The priority of the event which can take only the following values: “low”, “medium” or “high”"),
                    ],
                ),
                SchemaEntity(
                    label="Person",
                    properties=[
                        SchemaProperty(name="name", type="STRING"),
                        SchemaProperty(name="firstName", type="STRING"),
                        SchemaProperty(name="lastName", type="STRING"),
                        SchemaProperty(name="email", type="STRING"),
                        SchemaProperty(name="phone", type="STRING"),
                    ],
                ),
                SchemaEntity(
                    label="Place",
                    properties=[
                        SchemaProperty(name="name", type="STRING"),
                        SchemaProperty(name="description", type="STRING"),
                        SchemaProperty(name="address", type="STRING"),
                        SchemaProperty(name="latitude", type="FLOAT"),
                        SchemaProperty(name="longitude", type="FLOAT"),
                    ],
                ),
                SchemaEntity(
                    label="Project",
                    properties=[
                        SchemaProperty(name="name", type="STRING"),
                        SchemaProperty(name="description", type="STRING"),
                        SchemaProperty(name="statusProject", type="STRING", description="The status of the project which can take only the following values: “active” or “completed”"),
                        SchemaProperty(name="priority", type="STRING", description="The priority of the project which can take only the following values: “low”, “medium” or “high”"),
                        SchemaProperty(name="statDate", type="DATE"),
                        SchemaProperty(name="dueDate", type="DATE"),
                        SchemaProperty(name="goal", type="STRING"),
                    ],
                ),
                SchemaEntity(
                    label="Activity",
                    properties=[
                        SchemaProperty(name="name", type="STRING"),
                        SchemaProperty(name="description", type="STRING"),
                        SchemaProperty(name="statusActivity", type="STRING", description="The status of the activity which can take only the following values: “planned” or “completed”.”"),
                        SchemaProperty(name="priority", type="STRING", description="The priority of the activity which can take only the following values: “low”, “medium” or “high”."),
                        SchemaProperty(name="onDate", type="DATE"),
                        SchemaProperty(name="startTime", type="LOCAL_DATETIME"),
                        SchemaProperty(name="endTime", type="LOCAL_DATETIME"),
                    ],
                    description="An activity is a task or action that needs to be done, often expressed as a verb, frequently preceded by phrases like has to, needs to, will has to, required to, is going to.",
                ),
            ],
            "relations": [
                SchemaRelation(label="with"),
                SchemaRelation(label="occursAt"),
                SchemaRelation(label="projectActivity"),
                SchemaRelation(label="workOn"),
                SchemaRelation(label="doActivity"),
                SchemaRelation(label="partecipates")
            ],
            "potential_schema": [
                ("Person", "doActivity", "Activity"),
                ("Event", "with", "Person"),
                ("Person", "workOn", "Project"),
                ("Person", "partecipates", "Event"),
                ("Event", "occursAt", "Place"),
                ("Project", "projectActivity", "Activity"),          
            ],
        }
    
    
    
    
    
    schema2 = {
        "entities": [
            SchemaEntity(
            label="Person",
            description="""Represents a person involved in activities, events, or projects.""",
            properties=[
                SchemaProperty(name="name", type="STRING", description="""Full name of the person."""),
                SchemaProperty(name="email", type="STRING", description="""Email address of the person."""),
                SchemaProperty(name="phone", type="STRING", description="""Phone number of the person."""),
            ],
        ),
        SchemaEntity(
            label="Place",
            description="""A physical location where activities or events occur.""",
            properties=[
                SchemaProperty(name="name", type="STRING", description="""Name of the place."""),
                SchemaProperty(name="address", type="STRING", description="""Physical address of the place."""),
                SchemaProperty(name="latitude", type="FLOAT", description="""Latitude coordinate."""),
                SchemaProperty(name="longitude", type="FLOAT", description="""Longitude coordinate."""),
            ],
            
        ),
        SchemaEntity(
            label="Project",
            description="""A project represents a product or service that is to be produced.""",
            properties=[
                SchemaProperty(name="name", type="STRING", description="""Name of the project."""),
                SchemaProperty(name="description", type="STRING", description="""Brief description of the project."""),
                SchemaProperty(name="status", type="STRING", description="""Status of the project, values: active (default), completed."""),
                SchemaProperty(name="priority", type="STRING", description="""Priority level, values: low, medium, high. To be left blank if not indicated"""),
                SchemaProperty(name="dueDate", type="DATE", description="""Deadline for the project. To be defined only if specified directly in the text."""),
                SchemaProperty(name="goal", type="STRING", description="""Main objective or goal of the project."""),
            ],
        ),
        SchemaEntity(
            label="Activity",
            properties=[
                SchemaProperty(name="name", type="STRING", description="""Name or short description of the activity."""),
                SchemaProperty(name="description", type="STRING", description="""Detailed description of the activity."""),
                SchemaProperty(name="status", type="STRING", description="""Status of the activity, values: planned (default), completed."""),
                SchemaProperty(name="priority", type="STRING", description="""Priority level, values: low, medium (default), high."""),
                SchemaProperty(name="onDate", type="DATE", description="""Date when the activity occurs."""),
                SchemaProperty(name="startTime", type="LOCAL_DATETIME", description="""Start time of the activity."""),
            ],
            description="An activity is a task or action that needs to be done, often expressed as a verb, frequently preceded by phrases like has to, needs to, will has to, required to, is going to.",
        ),
        SchemaEntity(
            label="Event",
            description="""An event such as meetings or appointments""",
            properties=[
                SchemaProperty(name="name", type="STRING", description="""Name or title of the event."""),
                SchemaProperty(name="description", type="STRING", description="""Detailed description of the event."""),
                SchemaProperty(name="onDate", type="DATE", description="""Date of the event."""),
                SchemaProperty(name="startTime", type="LOCAL_DATETIME", description="""Start time of the event."""),
                SchemaProperty(name="endTime", type="LOCAL_DATETIME", description="""End time of the event."""),
                SchemaProperty(name="type", type="STRING", description="""Type of event, values: meeting, appointment, other."""),
                SchemaProperty(name="status", type="STRING", description="""Status of the event, values: planned (default), completed."""),
            ],
        ),
        SchemaEntity(
            label="RoutineActivity",
            description="""A recurring activity that happens at regular intervals.""",
            properties=[
                SchemaProperty(name="name", type="STRING", description="""Name or title of the recurring activity."""),
                SchemaProperty(name="description", type="STRING", description="""Detailed description of the recurring activity."""),
                SchemaProperty(name="frequency", type="STRING", description="""Frequency of the routine activity, values: daily, weekly, monthly, other."""),
                SchemaProperty(name="startDate", type="DATE", description="""Start date of the routine activity."""),
                SchemaProperty(name="endDate", type="DATE", description="""End date of the routine activity."""),
                SchemaProperty(name="atTime", type="LOCAL_TIME", description="""Time of day when the activity occurs."""),
            ]
        ),
        ],
        "relations": [
            SchemaRelation(label="doesActivity", description="Indicates that a person does an activity."),
            SchemaRelation(label="worksOn", description="Indicates that a person works on a project."),
            SchemaRelation(label="occursAt", description="Indicates that an activity or event occurs at a place."),
            SchemaRelation(label="projectActivity", description="Indicates the activities associated with a project"),
            SchemaRelation(label="partecipates", description="Indicates the participants of an event."),
            ],
        "potential_schema": [
            ("Person", "doesActivity", "Activity"),
            ("Person", "worksOn", "Project"),
            ("Activity", "occursAt", "Place"),
            ("Project", "projectActivity", "Activity"),
            ("Person", "partecipates", "Event"),
            ("Event", "occursAt", "Place"),
            ("Person", "doesActivity", "RoutineActivity"),
            ("RoutineActivity", "occursAt", "Place"),   

]        
    }
    
    #sostiutisci in user_input tutti gli a capo con lo spazioe
    input = user_input.replace("\n", " ")
    
    #Pipeline input
    pipe_inputs = {
        "splitter": {
            "text": input,
        },
        "schema":  schema2
    }
    
    
    #Execute the pipeline
    response = await pipe.run(pipe_inputs)
    driver.close()
    
    return response.result
    

    
async def resolve_entities():
        #Resolver da usare alla fine
    resolver = SinglePropertyExactMatchResolver(driver)
    res = await resolver.run()
    return res

