import logging
from typing import List, Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# A comprehensive list of prefixes needed for the TTL file.
TTL_PREFIXES = """
# ==================================================================
#  LingoMap v2.0 - AI Generated Semantic Blueprint
#  Generated on: {generation_date}
# ==================================================================

@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix mymap: <http://example.com/mapping-schema#> .
@prefix vocab: <http://example.com/vocab/> .
@prefix dcat: <http://www.w3.org/ns/dcat#> .
@prefix dcterms: <http://purl.org/dc/terms/> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .

# --- Standard Vocabularies ---
@prefix fibo-be-le-fbo: <https://spec.edmcouncil.org/fibo/ontology/BE/LegalEntities/FormalBusinessOrganizations/> .
@prefix fibo-fbc-fct-fse: <https://spec.edmcouncil.org/fibo/ontology/FBC/FunctionalEntities/FinancialServicesEntities/> .
@prefix fibo-fnd-plc-adr: <https://spec.edmcouncil.org/fibo/ontology/FND/Places/Addresses/> .
@prefix fibo-fnd-rel-rel: <https://spec.edmcouncil.org/fibo/ontology/FND/Relations/Relations/> .
@prefix fibo-fnd-dt-fd: <https://spec.edmcouncil.org/fibo/ontology/FND/DatesAndTimes/FinancialDates/> .
@prefix fibo-ind-ei-ei: <https://spec.edmcouncil.org/fibo/ontology/IND/EconomicIndicators/EconomicIndicators/> .
@prefix fibo-fbc-pas-caa: <https://spec.edmcouncil.org/fibo/ontology/FBC/ProductsAndServices/ClientsAndAccounts/> .
@prefix cmns-cls: <https://www.omg.org/spec/Commons/Classifiers/> .
@prefix cmns-loc: <https://www.omg.org/spec/Commons/Locations/> .
@prefix cmns-dt: <https://www.omg.org/spec/Commons/DatesAndTimes/> .
@prefix cmns-id: <https://www.omg.org/spec/Commons/Identifiers/> .
@prefix cmns-org: <https://www.omg.org/spec/Commons/Organizations/> .
@prefix lcc-lr: <https://www.omg.org/spec/LCC/Languages/LanguageRepresentation/> .
@prefix geo: <http://www.opengis.net/ont/geosparql#> .
@prefix sf: <http://www.opengis.net/ont/sf#> .
@prefix schema: <https://schema.org/> .

"""

def format_entities_to_ttl(entities: List[Dict[str, Any]]) -> str:
    """Formats the list of entities into TTL syntax."""
    if not entities:
        return ""
    
    ttl_blocks = []
    for entity in entities:
        if "error" in entity:
            continue
        block = f"""
<#{entity.get('entityId', 'UnknownEntity')}>
    a mymap:EntityMapping ;
    rdfs:label "{entity.get('entityLabel', 'No Label')}"@en ;
    rdfs:comment "{entity.get('entityComment', 'No comment.')}"@en ;
    mymap:mapsToClass {entity.get('mapsToClass', 'rdfs:Class')} .
"""
        ttl_blocks.append(block)
    
    return "\n# --- 1. Entity & Association Definitions ---\n" + "\n".join(ttl_blocks)

def format_associations_to_ttl(associations: List[Dict[str, Any]]) -> str:
    """Formats the list of relationships into TTL syntax."""
    if not associations:
        return ""
        
    ttl_blocks = []
    for assoc in associations:
        block = f"""
<#{assoc.get('associationId', 'unknown_link')}>
    a mymap:AssociationMapping ;
    rdfs:comment "{assoc.get('justification', 'No justification provided.')}"@en ;
    mymap:linksEntity <#{assoc.get('sourceEntity', 'SourceEntity')}> ;
    mymap:toEntity <#{assoc.get('targetEntity', 'TargetEntity')}> ;
    mymap:usingProperty {assoc.get('usingProperty', 'rdfs:seeAlso')} .
"""
        ttl_blocks.append(block)
    
    return "\n# --- 2. Association Mappings ---\n" + "\n".join(ttl_blocks)

def format_column_mappings_to_ttl(mappings: Dict[str, Any]) -> str:
    """Formats the dictionary of column mappings into TTL syntax."""
    if not mappings:
        return ""

    ttl_blocks = []
    for column_name, suggestion in mappings.items():
        if "error" in suggestion or not suggestion.get('mapsToProperty'):
            continue
        
        # Create a valid Turtle subject name from the column name
        map_name = f"map_{column_name.replace(' ', '_').replace('.', '_')}"
        
        # Choose the correct mapping class based on the suggestion
        mapping_class = suggestion.get('mappingType', 'ColumnMapping')
        if mapping_class not in ["ColumnMapping", "IdentifierMapping", "ClassificationMapping"]:
            mapping_class = "ColumnMapping"

        block = f"""
<#{map_name}>
    a mymap:{mapping_class} ;
    mymap:sourceColumn "{column_name}" ;
    mymap:mapsToProperty {suggestion.get('mapsToProperty', 'rdfs:label')} ;
    mymap:partOfEntity <#{suggestion.get('partOfEntity', 'UnknownEntity')}> ;
    mymap:confidenceScore "{suggestion.get('confidenceScore', 0.5)}"^^xsd:decimal ;
    rdfs:comment "{suggestion.get('justification', 'No justification.')}"@en .
"""
        ttl_blocks.append(block)

    return "\n# --- 3. Column to Property Mappings ---\n" + "\n".join(ttl_blocks)


def generate_semantic_blueprint_ttl(
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    column_mappings: Dict[str, Any]
) -> str:
    """
    Combines all structured data from the three stages into a single TTL file content.
    """
    logging.info("Generating final TTL semantic blueprint...")
    
    generation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Format prefixes
    prefixes_section = TTL_PREFIXES.format(generation_date=generation_date)
    
    # 2. Format entities
    entities_section = format_entities_to_ttl(entities)
    
    # 3. Format associations
    associations_section = format_associations_to_ttl(relationships)
    
    # 4. Format column mappings
    mappings_section = format_column_mappings_to_ttl(column_mappings)
    
    # 5. Combine all sections into the final TTL string
    final_ttl = (
        prefixes_section +
        entities_section +
        associations_section +
        mappings_section
    )
    
    logging.info("TTL blueprint generation complete.")
    return final_ttl