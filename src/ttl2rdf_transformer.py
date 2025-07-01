# ttl2rdf_transformer.py

import pandas as pd
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD, SKOS
from typing import Dict, Any
import logging
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LingoMapTransformer:
    """
    A generic, rule-driven engine to transform CSV data to RDF based on a LingoMap TTL blueprint.
    This version includes robust parsing and data type handling.
    """
    def __init__(self, base_uri: str = "http://example.com/data/"):
        self.base_uri = Namespace(base_uri)
        self.rules_g = Graph()
        self.output_g = Graph()
        self.mymap = Namespace('http://example.com/mapping-schema#')
        self.parsed_rules: Dict[str, Any] = {}
        self._bind_namespaces()

    def _bind_namespaces(self):
        """Binds all necessary namespaces to the output graph for a clean, complete output."""
        # Base and Common Prefixes
        self.output_g.bind("ex", self.base_uri)
        self.output_g.bind("rdfs", RDFS)
        self.output_g.bind("xsd", XSD)
        self.output_g.bind("mymap", self.mymap)
        self.output_g.bind("vocab", Namespace("http://example.com/vocab/"))
        self.output_g.bind("dcat", Namespace("http://www.w3.org/ns/dcat#"))
        self.output_g.bind("dcterms", Namespace("http://purl.org/dc/terms/"))
        self.output_g.bind("skos", SKOS)
        self.output_g.bind("schema", Namespace("https://schema.org/"))

        # FIBO Prefixes
        self.output_g.bind("fibo-be-le-fbo", Namespace("https://spec.edmcouncil.org/fibo/ontology/BE/LegalEntities/FormalBusinessOrganizations/"))
        self.output_g.bind("fibo-fbc-fct-fse", Namespace("https://spec.edmcouncil.org/fibo/ontology/FBC/FunctionalEntities/FinancialServicesEntities/"))
        self.output_g.bind("fibo-fnd-plc-adr", Namespace("https://spec.edmcouncil.org/fibo/ontology/FND/Places/Addresses/"))
        self.output_g.bind("fibo-fnd-rel-rel", Namespace("https://spec.edmcouncil.org/fibo/ontology/FND/Relations/Relations/"))
        self.output_g.bind("fibo-fnd-dt-fd", Namespace("https://spec.edmcouncil.org/fibo/ontology/FND/DatesAndTimes/FinancialDates/"))
        self.output_g.bind("fibo-ind-ei-ei", Namespace("https://spec.edmcouncil.org/fibo/ontology/IND/EconomicIndicators/EconomicIndicators/"))
        self.output_g.bind("fibo-fbc-pas-caa", Namespace("https://spec.edmcouncil.org/fibo/ontology/FBC/ProductsAndServices/ClientsAndAccounts/"))
        self.output_g.bind("fibo-fnd-org-org", Namespace("https://spec.edmcouncil.org/fibo/ontology/FND/Organizations/Organizations/"))
        self.output_g.bind("fibo-be-oac-cctl", Namespace("https://spec.edmcouncil.org/fibo/ontology/BE/OwnershipAndControl/CorporateControl/"))
        self.output_g.bind("fibo-fnd-utl-alx", Namespace("https://spec.edmcouncil.org/fibo/ontology/FND/Utilities/Analytics/"))

        # OMG Commons Prefixes
        self.output_g.bind("cmns-cls", Namespace("https://www.omg.org/spec/Commons/Classifiers/"))
        self.output_g.bind("cmns-col", Namespace("https://www.omg.org/spec/Commons/Collections/"))
        self.output_g.bind("cmns-loc", Namespace("https://www.omg.org/spec/Commons/Locations/"))
        self.output_g.bind("cmns-dt", Namespace("https://www.omg.org/spec/Commons/DatesAndTimes/"))
        self.output_g.bind("cmns-id", Namespace("https://www.omg.org/spec/Commons/Identifiers/"))
        self.output_g.bind("cmns-org", Namespace("https://www.omg.org/spec/Commons/Organizations/"))
        self.output_g.bind("cmns-txt", Namespace("https://www.omg.org/spec/Commons/TextDatatype/"))
        
        # LCC Prefix
        self.output_g.bind("lcc-lr", Namespace("https://www.omg.org/spec/LCC/Languages/LanguageRepresentation/"))

        # GeoSPARQL Prefixes
        self.output_g.bind("geo", Namespace("http://www.opengis.net/ont/geosparql#"))
        self.output_g.bind("geor", Namespace("http://www.opengis.net/def/rule/geosparql/"))
        self.output_g.bind("geof", Namespace("http://www.opengis.net/def/function/geosparql/"))
        self.output_g.bind("sf", Namespace("http://www.opengis.net/ont/sf#"))

    def _pre_parse_rules(self):
        """
        Parses the rules graph once and stores instructions in a structured dictionary.
        """
        logging.info("Pre-parsing mapping rules for efficient transformation...")
        self.parsed_rules = {
            "entities": {}, "associations": [], "column_mappings": {}
        }

        # Parse Entities
        for entity_node in self.rules_g.subjects(RDF.type, self.mymap.EntityMapping):
            entity_id = str(entity_node).split('#')[-1]
            self.parsed_rules['entities'][entity_id] = {
                'class': self.rules_g.value(entity_node, self.mymap.mapsToClass)
            }

        # Parse Associations
        for assoc_node in self.rules_g.subjects(RDF.type, self.mymap.AssociationMapping):
            source_node = self.rules_g.value(assoc_node, self.mymap.linksEntity)
            target_node = self.rules_g.value(assoc_node, self.mymap.toEntity)
            self.parsed_rules['associations'].append({
                'source': str(source_node).split('#')[-1] if source_node else None,
                'target': str(target_node).split('#')[-1] if target_node else None,
                'property': self.rules_g.value(assoc_node, self.mymap.usingProperty)
            })

        # Parse Column Mappings, now including data type
        for mapping_node in self.rules_g.subjects(RDF.type, self.mymap.ColumnMapping):
            col_name = str(self.rules_g.value(mapping_node, self.mymap.sourceColumn))
            part_of_node = self.rules_g.value(mapping_node, self.mymap.partOfEntity)
            self.parsed_rules['column_mappings'][col_name] = {
                'property': self.rules_g.value(mapping_node, self.mymap.mapsToProperty),
                'partOf': str(part_of_node).split('#')[-1] if part_of_node else None,
                'datatype': self.rules_g.value(mapping_node, self.mymap.hasDataType) # Capture the datatype
            }
        
        logging.info(f"Rules parsed: {len(self.parsed_rules['entities'])} entities, {len(self.parsed_rules['associations'])} associations, {len(self.parsed_rules['column_mappings'])} column mappings.")

    def _get_stable_id(self, row: pd.Series, entity_id: str) -> str:
        """Creates a stable and unique ID for an entity based on row content."""
        # Use a primary key if the entity is central, like a Bank Branch
        if entity_id == 'BankBranchEntity' and 'UNINUM' in row:
            return str(row['UNINUM'])
        if entity_id == 'BankInstitutionEntity' and 'CERT' in row:
            return str(row['CERT'])
        
        # For composite entities like Address, create a hash from its components
        if entity_id == 'PhysicalAddressEntity':
            address_parts = "".join(str(row.get(col, '')) for col in ['ADDRESS', 'CITY', 'ZIP'])
            return hashlib.md5(address_parts.encode()).hexdigest()

        # Fallback to a generic row-based ID
        return str(row.get('UNINUM', row.name))

    def _process_row(self, row: pd.Series):
        """Processes a single row of the DataFrame according to the parsed rules."""
        row_entities: Dict[str, URIRef] = {}

        # 1. Create all defined entities for this row
        for entity_id, entity_info in self.parsed_rules['entities'].items():
            stable_id = self._get_stable_id(row, entity_id)
            entity_uri = self.base_uri[f"{entity_id}_{stable_id}"]
            row_entities[entity_id] = entity_uri
            self.output_g.add((entity_uri, RDF.type, entity_info['class']))

        # 2. Apply Column Mappings with Datatype Handling
        for col_name, mapping_info in self.parsed_rules['column_mappings'].items():
            if col_name in row.index and pd.notna(row.at[col_name]):
                value = row.at[col_name]
                target_entity_id = mapping_info['partOf']
                subject_uri = row_entities.get(target_entity_id)
                predicate_uri = mapping_info['property']
                datatype_uri = mapping_info.get('datatype') # Get the datatype URI
                
                if subject_uri and predicate_uri:
                    # Apply the specific datatype if it exists in the rules
                    literal = Literal(value, datatype=datatype_uri) if datatype_uri else Literal(value)
                    self.output_g.add((subject_uri, predicate_uri, literal))

        # 3. Apply Associations
        for assoc_info in self.parsed_rules['associations']:
            source_uri = row_entities.get(assoc_info['source'])
            target_uri = row_entities.get(assoc_info['target'])
            predicate_uri = assoc_info['property']
            if source_uri and target_uri and predicate_uri:
                self.output_g.add((source_uri, predicate_uri, target_uri))

        # 4. Apply Conditional Rules (Example)
        if 'MAINOFF' in row.index and row.at['MAINOFF'] == 1:
            branch_uri = row_entities.get('BankBranchEntity')
            if branch_uri:
                headquarters_class = URIRef("https://spec.edmcouncil.org/fibo/ontology/FND/Organizations/Organizations/Headquarters")
                self.output_g.add((branch_uri, RDF.type, headquarters_class))

    def transform(self, df: pd.DataFrame, ttl_content: str) -> Graph:
        """
        Main transformation method.
        
        Args:
            df: The pandas DataFrame containing the source data.
            ttl_content: A string containing the TTL mapping rules.
            
        Returns:
            An rdflib Graph object with the transformed data.
        """
        try:
            self.rules_g.parse(data=ttl_content, format='turtle')
        except Exception as e:
            logging.error(f"❌ Failed to parse TTL rules content: {e}")
            raise ValueError(f"Invalid TTL Rules: {e}")

        self._pre_parse_rules()
        
        logging.info(f"🔄 Starting transformation of {len(df)} rows...")
        for _, row in df.iterrows():
            self._process_row(row)

        logging.info("✅ Data transformation completed.")
        
        # --- THIS IS THE FIX ---
        # Return the graph object directly, not a serialized string.
        return self.output_g
        # --- END OF FIX ---