from rdflib import Graph, URIRef, RDF, RDFS, OWL
from neo4j_graphrag.experimental.components.schema import (
    SchemaBuilder,
    SchemaEntity,
    SchemaProperty,
    SchemaRelation,
)

XSD_TYPE_MAPPING = {
    "xsd:string": "STRING",
    "xsd:integer": "INTEGER",
    "xsd:float": "FLOAT",
    "xsd:boolean": "BOOLEAN",
    "xsd:date": "DATE",
    "xsd:dateTime": "LOCAL_DATETIME",
}

def parse_ontology(ontology_file):
    """
    Parses an ontology in Turtle format and converts it into SchemaEntity and SchemaRelation objects.
    Ensures subclasses inherit properties from their parent classes and includes descriptions.
    """
    g = Graph()
    g.parse(ontology_file, format="turtle")

    # Define RDF/OWL types
    OWL_CLASS = URIRef("http://www.w3.org/2002/07/owl#Class")
    OWL_OBJECT_PROPERTY = URIRef("http://www.w3.org/2002/07/owl#ObjectProperty")
    OWL_DATATYPE_PROPERTY = URIRef("http://www.w3.org/2002/07/owl#DatatypeProperty")

    # Extract entities (classes)
    entities_dict = {}
    for s in g.subjects(RDF.type, OWL_CLASS):
        label = g.value(s, RDFS.label, default=None)
        if label:
            label = str(label)
            entities_dict[s] = SchemaEntity(label=label, properties=[])

    # Extract subclass relationships
    subclass_map = {}
    for s, _, o in g.triples((None, RDFS.subClassOf, None)):
        if s in entities_dict and o in entities_dict:
            subclass_map[s] = o  # s (subclass) inherits from o (parent class)

    # Extract properties and associate them with entities
    property_map = {}
    for s, _, _ in g.triples((None, RDF.type, OWL_DATATYPE_PROPERTY)):
        domain = g.value(s, RDFS.domain)
        label = g.value(s, RDFS.label, default=None)
        description = g.value(s, RDFS.comment, default=None)
        xsd_type = g.value(s, RDFS.range)
        
        prop_type = XSD_TYPE_MAPPING.get(xsd_type, "STRING")  # Default to STRING

        if domain and label and domain in entities_dict:
            prop = SchemaProperty(
                name=str(label),
                type=prop_type,
                description=str(description) if description else None
            )
            entities_dict[domain].properties.append(prop)
            property_map[s] = prop  # Store property for inheritance

    # Inherit properties from parent classes
    for subclass, parent in subclass_map.items():
        if subclass in entities_dict and parent in entities_dict:
            entities_dict[subclass].properties.extend(entities_dict[parent].properties)

    # Extract relationships (object properties)
    relations = []
    potential_schema = []
    for s, _, _ in g.triples((None, RDF.type, OWL_OBJECT_PROPERTY)):
        domain = g.value(s, RDFS.domain)
        range_ = g.value(s, RDFS.range)
        label = g.value(s, RDFS.label, default=None)
        description = g.value(s, RDFS.comment, default=None)

        if label:
            relations.append(SchemaRelation(
                label=str(label),
                description=str(description) if description else None
            ))

            if domain and range_ and domain in entities_dict and range_ in entities_dict:
                potential_schema.append((entities_dict[domain].label, str(label), entities_dict[range_].label))

    # Format the extracted schema using SchemaBuilder
    schema =  {
        "entities": list(entities_dict.values()),
        "relations": relations,
        "potential_schema": potential_schema
    }

    return schema


if __name__ == "__main__":
    ontology_path = "models/TAMOntology.ttl"  
    schema = parse_ontology(ontology_path)
    print(schema)
