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
@prefix fibo-fnd-org-org: <https://spec.edmcouncil.org/fibo/ontology/FND/Organizations/Organizations/> .
@prefix fibo-be-oac-cctl: <https://spec.edmcouncil.org/fibo/ontology/BE/OwnershipAndControl/CorporateControl/> .
@prefix fibo-fnd-utl-alx: <https://spec.edmcouncil.org/fibo/ontology/FND/Utilities/Analytics/> .
@prefix cmns-cls: <https://www.omg.org/spec/Commons/Classifiers/> .
@prefix cmns-col: <https://www.omg.org/spec/Commons/Collections/> .
@prefix cmns-loc: <https://www.omg.org/spec/Commons/Locations/> .
@prefix cmns-dt: <https://www.omg.org/spec/Commons/DatesAndTimes/> .
@prefix cmns-id: <https://www.omg.org/spec/Commons/Identifiers/> .
@prefix cmns-org: <https://www.omg.org/spec/Commons/Organizations/> .
@prefix cmns-txt: <https://www.omg.org/spec/Commons/TextDatatype/> .
@prefix lcc-lr: <https://www.omg.org/spec/LCC/Languages/LanguageRepresentation/> .
@prefix geo: <http://www.opengis.net/ont/geosparql#> .
@prefix geor: <http://www.opengis.net/def/rule/geosparql/> .
@prefix geof: <http://www.opengis.net/def/function/geosparql/> .
@prefix sf: <http://www.opengis.net/ont/sf#> .
@prefix schema: <https://schema.org/> .

"""

def format_rdf_term(term: str) -> str:
    """
    智能地格式化一個詞彙，使其符合 Turtle RDF 語法。
    - 如果是完整 URI，則用 <...> 包圍。
    - 如果是前綴名或字面值，則保持不變。
    """
    if isinstance(term, str) and term.startswith(('http://', 'https://')):
        return f"<{term}>"
    # 對於 "rdfs:Class" 這樣的前綴名，或者其他情況，直接返回
    return term

def format_conditional_rules_to_ttl(mappings: Dict[str, Any]) -> str:
    """Formats conditional rules into human-readable TTL comments."""
    ttl_blocks = []
    # 過濾出所有被識別為規則的建議
    rules = [m for m in mappings.values() if m.get("isRule")]
    
    if not rules:
        return ""

    for rule in rules:
        condition = rule.get('condition', {})
        action = rule.get('action', {})
        block = f"""
# --- Conditional Rule: {rule.get('ruleId', 'N/A')} ---
# @description: {rule.get('justification', '')}
# @if-column: "{rule.get('sourceColumn')}"
# @if-operator: "{condition.get('operator')}"
# @if-value: "{condition.get('value')}"
# @then-action: "{action.get('type')}"
# @then-value: <{action.get('value')}>
# @on-entity: <#{rule.get('targetEntity')}>
"""
        ttl_blocks.append(block)
    
    return "\n# --- 4. Conditional Transformation Rules (for ETL tool) ---\n" + "\n".join(ttl_blocks)

def format_entities_to_ttl(entities: List[Dict[str, Any]]) -> str:
    """Formats the list of entities into TTL syntax."""
    if not entities:
        return ""
    
    ttl_blocks = []
    for entity in entities:
        if "error" in entity:
            continue
        
        maps_to_class_formatted = format_rdf_term(entity.get('mapsToClass', 'rdfs:Class'))

        block = f"""
<#{entity.get('entityId', 'UnknownEntity')}>
    a mymap:EntityMapping ;
    rdfs:label "{entity.get('entityLabel', 'No Label')}"@en ;
    rdfs:comment "{entity.get('entityComment', 'No comment.')}"@en ;
    mymap:mapsToClass {maps_to_class_formatted} .
"""
        ttl_blocks.append(block)
    
    return "\n# --- 1. Entity & Association Definitions ---\n" + "\n".join(ttl_blocks)

def format_associations_to_ttl(associations: List[Dict[str, Any]]) -> str:
    """Formats the list of relationships into TTL syntax."""
    if not associations:
        return ""
        
    ttl_blocks = []
    for assoc in associations:
        # 使用新的輔助函式來格式化 usingProperty
        using_property_formatted = format_rdf_term(assoc.get('usingProperty', 'rdfs:seeAlso'))

        block = f"""
<#{assoc.get('associationId', 'unknown_link')}>
    a mymap:AssociationMapping ;
    rdfs:comment "{assoc.get('justification', 'No justification provided.')}"@en ;
    mymap:linksEntity <#{assoc.get('sourceEntity', 'SourceEntity')}> ;
    mymap:toEntity <#{assoc.get('targetEntity', 'TargetEntity')}> ;
    mymap:usingProperty {using_property_formatted} .
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
            
        # 使用新的輔助函式來格式化 mapsToProperty
        maps_to_property_formatted = format_rdf_term(suggestion.get('mapsToProperty', 'rdfs:label'))

        block = f"""
<#{map_name}>
    a mymap:{mapping_class} ;
    mymap:sourceColumn "{column_name}" ;
    mymap:mapsToProperty {maps_to_property_formatted} ;
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
    
    # 4.5 Format conditional rules
    rules_section = format_conditional_rules_to_ttl(column_mappings)
    
    # 5. Combine all sections into the final TTL string
    property_mappings = {k: v for k, v in column_mappings.items() if not v.get("isRule")}
    mappings_section = format_column_mappings_to_ttl(property_mappings)
    final_ttl = (
        prefixes_section +
        entities_section +
        associations_section +
        mappings_section +
        rules_section
    )
    
    logging.info("TTL blueprint generation complete.")
    return final_ttl