from rdflib import Graph, URIRef, RDF, RDFS, OWL
from neo4j_graphrag.experimental.components.schema import (
    SchemaEntity,
    SchemaProperty,
    SchemaRelation,
)

XSD_TYPE_MAPPING = {
    "http://www.w3.org/2001/XMLSchema#string": "STRING",
    "http://www.w3.org/2001/XMLSchema#normalizedString": "STRING",
    "http://www.w3.org/2001/XMLSchema#token": "STRING",

    "http://www.w3.org/2001/XMLSchema#integer": "INTEGER",
    "http://www.w3.org/2001/XMLSchema#int": "INTEGER",
    "http://www.w3.org/2001/XMLSchema#long": "INTEGER",
    "http://www.w3.org/2001/XMLSchema#short": "INTEGER",
    "http://www.w3.org/2001/XMLSchema#byte": "INTEGER",
    "http://www.w3.org/2001/XMLSchema#nonNegativeInteger": "INTEGER",

    "http://www.w3.org/2001/XMLSchema#float": "FLOAT",
    "http://www.w3.org/2001/XMLSchema#decimal": "FLOAT",
    "http://www.w3.org/2001/XMLSchema#double": "FLOAT",

    "http://www.w3.org/2001/XMLSchema#boolean": "BOOLEAN",

    "http://www.w3.org/2001/XMLSchema#date": "DATE",
    "http://www.w3.org/2001/XMLSchema#dateTime": "LOCAL_DATETIME",
    "http://www.w3.org/2001/XMLSchema#time": "LOCAL_TIME",
    "http://www.w3.org/2001/XMLSchema#duration": "DURATION",
}

def parse_ontology(ontology_file):
    g = Graph()
    g.parse(ontology_file, format="turtle")

    OWL_CLASS = URIRef("http://www.w3.org/2002/07/owl#Class")
    OWL_OBJECT_PROPERTY = URIRef("http://www.w3.org/2002/07/owl#ObjectProperty")
    OWL_DATATYPE_PROPERTY = URIRef("http://www.w3.org/2002/07/owl#DatatypeProperty")

    entities_dict = {}
    subclass_map = {}

    # Estrai entità (classi)
    for s in g.subjects(RDF.type, OWL_CLASS):
        label = g.value(s, RDFS.label, default=None)
        description = g.value(s, RDFS.comment, default=None)
        if label and description:
            entities_dict[s] = SchemaEntity(label=str(label), properties=[], description=str(description))

    # Estrai sottoclassi
    for s, _, o in g.triples((None, RDFS.subClassOf, None)):
        if s in entities_dict and o in entities_dict:
            subclass_map.setdefault(o, []).append(s)

    # Estrai proprietà datatype
    property_map = {}
    for s, _, _ in g.triples((None, RDF.type, OWL_DATATYPE_PROPERTY)):
        domain = g.value(s, RDFS.domain)
        label = g.value(s, RDFS.label, default=None)
        description = g.value(s, RDFS.comment, default=None)
        xsd_type = g.value(s, RDFS.range)
        prop_type = XSD_TYPE_MAPPING.get(str(xsd_type), "STRING")

        if domain and label and domain in entities_dict:
            prop = SchemaProperty(
                name=str(label),
                type=prop_type,
                description=str(description) if description else None
            )
            entities_dict[domain].properties.append(prop)
            property_map[s] = prop

            # Eredita anche alle sottoclassi
            for subclass in subclass_map.get(domain, []):
                entities_dict[subclass].properties.append(prop)

    # Estrai relazioni (object properties)
    relations = []
    potential_schema = []

    for s, _, _ in g.triples((None, RDF.type, OWL_OBJECT_PROPERTY)):
        domain = g.value(s, RDFS.domain)
        range_ = g.value(s, RDFS.range)
        label = g.value(s, RDFS.label, default=None)
        description = g.value(s, RDFS.comment, default=None)

        if label:
            relation_name = str(label)
            relations.append(SchemaRelation(label=relation_name, description=str(description) if description else None))

            # Estendi dominio e range con le sottoclassi
            domains = [domain] + subclass_map.get(domain, []) if domain else []
            ranges = [range_] + subclass_map.get(range_, []) if range_ else []

            for d in domains:
                for r in ranges:
                    if d in entities_dict and r in entities_dict:
                        potential_schema.append((
                            entities_dict[d].label,
                            relation_name,
                            entities_dict[r].label
                        ))

    schema = {
        "entities": list(entities_dict.values()),
        "relations": relations,
        "potential_schema": potential_schema
    }
    
    return schema
    
    
def print_schema(schema):
    for entity in schema["entities"]:
        print(f"Entity: {entity.label}")
        print(f"  Description: {entity.description}")
        for prop in entity.properties:
            print(f"  Property: {prop.name} ({prop.type})")
            print(f"    Description: {prop.description}")

    for relation in schema["relations"]:
        print(f"Relation: {relation.label}")
        print(f"  Description: {relation.description}")

    for potential in schema["potential_schema"]:
        print(f"Potential Schema: {potential[0]} - {potential[1]} - {potential[2]}")


if __name__ == "__main__":
    ontology_path = "models/TAMOntology.ttl"  
    schema = parse_ontology(ontology_path)

