"""
LingoMap Transformation Script
Reads a CSV file and a set of mapping rules in Turtle format,
then generates the final RDF data.
"""
# python ttl2rdf_transformer.py lingomap_rules.ttl FDIC_Insured_Banks.csv fdic_banks_output.xml

import sys
from typing import Dict
import pandas as pd
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, XSD, SKOS


class LingoMapTransformer:
    """
    Transforms CSV data to RDF based on LingoMap mapping rules.
    """

    def __init__(self, rules_file: str, csv_file: str, output_file: str, base_uri: str = "http://example.com/data/"):
        self.rules_file = rules_file
        self.csv_file = csv_file
        self.output_file = output_file
        self.base_uri = Namespace(base_uri)

        self.rules_g = Graph()
        self.output_g = Graph()

        self.mymap = Namespace('http://example.com/mapping-schema#')
        self.vocab = Namespace('http://example.com/vocab/')

        # Pre-bind namespaces for cleaner output
        self.bind_namespaces()

    def bind_namespaces(self):
        """Binds all necessary namespaces to the output graph."""
        self.output_g.bind("rdfs", RDFS)
        self.output_g.bind("xsd", XSD)
        self.output_g.bind("skos", SKOS)
        self.output_g.bind("dcat", Namespace("http://www.w3.org/ns/dcat#"))
        self.output_g.bind("dcterms", Namespace("http://purl.org/dc/terms/"))
        self.output_g.bind("fibo-be-le-fbo", Namespace(
            "https://spec.edmcouncil.org/fibo/ontology/BE/LegalEntities/FormalBusinessOrganizations/"))
        self.output_g.bind("fibo-fnd-plc-adr", Namespace(
            "https://spec.edmcouncil.org/fibo/ontology/FND/Places/Addresses/"))
        self.output_g.bind("fibo-fnd-rel-rel", Namespace(
            "https://spec.edmcouncil.org/fibo/ontology/FND/Relations/Relations/"))
        self.output_g.bind("fibo-fnd-org-org", Namespace(
            "https://spec.edmcouncil.org/fibo/ontology/FND/Organizations/Organizations/"))
        self.output_g.bind("fibo-fbc-fct-fse", Namespace(
            "https://spec.edmcouncil.org/fibo/ontology/FBC/FunctionalEntities/NorthAmericanFunctionalEntities/USFinancialServicesEntities/"))
        self.output_g.bind("fibo-ind-ei-ei", Namespace(
            "https://spec.edmcouncil.org/fibo/ontology/IND/EconomicIndicators/EconomicIndicators/"))
        self.output_g.bind(
            "cmns-loc", Namespace("https://www.omg.org/spec/Commons/Locations/"))
        self.output_g.bind(
            "cmns-dt", Namespace("https://www.omg.org/spec/Commons/DatesAndTimes/"))
        self.output_g.bind(
            "cmns-id", Namespace("https://www.omg.org/spec/Commons/Identifiers/"))
        self.output_g.bind(
            "lcc-lr", Namespace("https://www.omg.org/spec/LCC/Languages/LanguageRepresentation/"))
        self.output_g.bind(
            "cmns-org", Namespace("https://www.omg.org/spec/Commons/Organizations/"))
        self.output_g.bind("geo", Namespace(
            "http://www.opengis.net/ont/geosparql#"))
        self.output_g.bind("sf", Namespace("http://www.opengis.net/ont/sf#"))
        self.output_g.bind("schema", Namespace("https://schema.org/"))
        self.output_g.bind("vocab", self.vocab)

    def load_and_parse_rules(self):
        """Loads and parses the mapping rules from the Turtle file."""
        print("🔍 Load and parse mapping rules from the Turtle file...")
        self.rules_g.parse(self.rules_file, format='turtle')
        print(f"✅ Successfully parsed {len(self.rules_g)} triples.")

    def generate_skos_vocabularies(self, df: pd.DataFrame):
        """Dynamically generates SKOS vocabularies from the data."""
        print("🏷️ Generate SKOS concept vocabularies...")
        for rule_node in self.rules_g.subjects(RDF.type, self.mymap.ConceptGenerationMapping):
            label_col = self.rules_g.value(rule_node, self.mymap.labelColumn)
            notation_col = self.rules_g.value(
                rule_node, self.mymap.notationColumn)
            id_prefix = self.rules_g.value(
                rule_node, self.mymap.conceptIdPrefix)
            scheme_label = self.rules_g.value(
                rule_node, self.mymap.schemeLabel)

            if not (label_col and notation_col and id_prefix):
                continue

            try:
                # Ensure the fields exist in the DataFrame
                label_col_str = str(label_col)
                notation_col_str = str(notation_col)

                if label_col_str not in df.columns or notation_col_str not in df.columns:
                    print(
                        f"⚠️ Warning: Field {label_col_str} or {notation_col_str} not found, skipping this concept mapping")
                    continue

                # Use a safer way to process data
                for index, row in df.iterrows():
                    try:
                        # Get the values and check
                        label_val = row[label_col_str]
                        notation_val = row[notation_col_str]

                        # Check if it's a missing value (using pandas' isna function)
                        if pd.isna(label_val) or pd.isna(notation_val):
                            continue

                        # Convert to string
                        label_str = str(label_val)
                        notation_str = str(notation_val)

                        # Avoid duplicate concepts
                        concept_uri = self.vocab[f"{id_prefix}{notation_str}"]

                        # Check if this concept has already been added
                        if (concept_uri, RDF.type, SKOS.Concept) not in self.output_g:
                            self.output_g.add(
                                (concept_uri, RDF.type, SKOS.Concept))
                            self.output_g.add(
                                (concept_uri, SKOS.prefLabel, Literal(label_str, lang="en")))
                            self.output_g.add(
                                (concept_uri, SKOS.notation, Literal(notation_str)))

                    except Exception as e:
                        # Skip rows with problems, continue processing the next row
                        continue

            except Exception as e:
                print(f"⚠️ Warning: Error processing concept mapping: {e}")
                continue

        print("✅ SKOS concept vocabularies generated.")

    def process_row(self, row: pd.Series, rules: Dict):
        """Processes a single row of the CSV data."""
        cert = row.get('CERT')
        uninum = row.get('UNINUM')

        if pd.isna(cert) or pd.isna(uninum):
            return

        inst_uri = self.base_uri[f"institution/{int(cert)}"]
        branch_uri = self.base_uri[f"branch/{int(uninum)}"]

        # --- Fix: Use a simple string as a key to find ---
        self.output_g.add((inst_uri, RDF.type, rules['entities']['BankInstitutionEntity']))
        
        # Process BankBranchEntity which may have multiple classes
        branch_classes = rules['entities']['BankBranchEntity']
        if isinstance(branch_classes, list):
            for branch_class in branch_classes:
                self.output_g.add((branch_uri, RDF.type, branch_class))
        else:
            self.output_g.add((branch_uri, RDF.type, branch_classes))

        if row.get('MAINOFF') == 1:
            self.output_g.add((branch_uri, RDF.type, Namespace("https://spec.edmcouncil.org/fibo/ontology/FND/Organizations/Organizations/").Headquarters))

        entities = {'BankInstitutionEntity': inst_uri, 'BankBranchEntity': branch_uri}

        # --- Fix: Association logic ---
        for assoc_rule in rules['associations'].values():
            from_entity_uri = entities.get(assoc_rule['linksEntity'])
            to_entity_key = assoc_rule['toEntity']

            create_assoc = True
            if 'CSA_Entity' in str(to_entity_key) and row.get('CSA_FLG') != 1:
                create_assoc = False
            if 'CBSA_Entity' in str(to_entity_key) and pd.isna(row.get('CBSA')):
                create_assoc = False

            if from_entity_uri and create_assoc:
                bnode = BNode()
                self.output_g.add((from_entity_uri, assoc_rule['usingProperty'], bnode))
                
                # Process the type of toEntity
                to_entity_classes = rules['entities'].get(to_entity_key)
                if to_entity_classes:
                    if isinstance(to_entity_classes, list):
                        for cls in to_entity_classes:
                            self.output_g.add((bnode, RDF.type, cls))
                    else:
                        self.output_g.add((bnode, RDF.type, to_entity_classes))

                entities[to_entity_key] = bnode

                if 'CBSA_Entity' in str(to_entity_key):
                    if row.get('CBSA_METRO_FLG') == 1:
                        self.output_g.add((bnode, RDF.type, Namespace("https://spec.edmcouncil.org/fibo/ontology/IND/EconomicIndicators/EconomicIndicators/").MetropolitanStatisticalArea))
                    elif row.get('CBSA_MICRO_FLG') == 1:
                        self.output_g.add((bnode, RDF.type, Namespace("https://spec.edmcouncil.org/fibo/ontology/IND/EconomicIndicators/EconomicIndicators/").MicropolitanStatisticalArea))
        
        # --- Fix: Field mapping logic ---
        for col_rule in rules['column_mappings'].values():
            value = row.get(col_rule['sourceColumn'])
            if pd.notna(value):
                target_entity_uri = entities.get(col_rule['partOf'])
                if target_entity_uri:
                    # Special handling for date types
                    if col_rule.get('hasDataType') == XSD.date:
                        try:
                            # Try to convert date format
                            if isinstance(value, str) and value.strip():
                                # If it's an Excel date serial number (e.g. 42539)
                                if value.isdigit():
                                    # Convert Excel serial number to date (need extra processing)
                                    # Skip this format for now
                                    continue
                                # If it's other formats, skip for now
                                continue
                            else:
                                continue
                        except:
                            # If date conversion fails, skip this field
                            continue
                    else:
                        # Non-date fields are processed normally
                        literal = Literal(value, datatype=col_rule.get('hasDataType'))
                        self.output_g.add((target_entity_uri, col_rule['mapsToProperty'], literal))
        
        # --- Fix: Transformation mapping logic ---
        for trans_rule in rules['transformation_mappings'].values():
            target_entity_uri = entities.get(trans_rule['partOf'])
            if target_entity_uri:
                template = trans_rule['transformationTemplate']
                try:
                    # Ensure all input field values exist
                    if all(pd.notna(row.get(col)) for col in trans_rule['inputColumns']):
                        formatted_str = template.format(**{col: row[col] for col in trans_rule['inputColumns']})
                        literal = Literal(formatted_str, datatype=trans_rule.get('hasDataType'))
                        self.output_g.add((target_entity_uri, trans_rule['mapsToProperty'], literal))
                except KeyError:
                    pass

    def pre_parse_rules(self) -> Dict:
        """Pre-parses rules for faster access during transformation."""
        print("⚙️ Pre-parse rules for faster access during transformation...")
        rules = {
            'entities': {},
            'associations': {},
            'column_mappings': {},
            'transformation_mappings': {},
            'identifier_mappings': {}
        }

        # Parse Entity Mappings
        for s, _, o in self.rules_g.triples((None, self.mymap.mapsToClass, None)):
            # --- Fix: Use the fragment after # as the key ---
            entity_key = str(s).split('#')[-1]

            # Handle multiple classes for one entity
            if entity_key in rules['entities']:
                if isinstance(rules['entities'][entity_key], list):
                    rules['entities'][entity_key].append(o)
                else:
                    rules['entities'][entity_key] = [
                        rules['entities'][entity_key], o]
            else:
                rules['entities'][entity_key] = o

        # Parse Association Mappings
        for s in self.rules_g.subjects(RDF.type, self.mymap.AssociationMapping):
            # --- Fix: Use the fragment after # as the key ---
            assoc_key = str(s).split('#')[-1]
            rules['associations'][assoc_key] = {
                'linksEntity': str(self.rules_g.value(s, self.mymap.linksEntity)).split('#')[-1],
                'toEntity': str(self.rules_g.value(s, self.mymap.toEntity)).split('#')[-1],
                'usingProperty': self.rules_g.value(s, self.mymap.usingProperty)
            }

        # Parse Column Mappings
        for s in self.rules_g.subjects(RDF.type, self.mymap.ColumnMapping):
            # --- Fix: Use the fragment after # as the key ---
            col_key = str(s).split('#')[-1]
            rules['column_mappings'][col_key] = {
                'sourceColumn': str(self.rules_g.value(s, self.mymap.sourceColumn)),
                'mapsToProperty': self.rules_g.value(s, self.mymap.mapsToProperty),
                'partOf': str(self.rules_g.value(s, self.mymap.partOf)).split('#')[-1],
                'hasDataType': self.rules_g.value(s, self.mymap.hasDataType)
            }

        # Parse Transformation Mappings
        for s in self.rules_g.subjects(RDF.type, self.mymap.TransformationMapping):
            # --- Fix: Use the fragment after # as the key ---
            trans_key = str(s).split('#')[-1]
            rules['transformation_mappings'][trans_key] = {
                'partOf': str(self.rules_g.value(s, self.mymap.partOf)).split('#')[-1],
                'inputColumns': [str(c) for c in self.rules_g.objects(s, self.mymap.inputColumns)],
                'transformationTemplate': str(self.rules_g.value(s, self.mymap.transformationTemplate)),
                'mapsToProperty': self.rules_g.value(s, self.mymap.mapsToProperty),
                'hasDataType': self.rules_g.value(s, self.mymap.hasDataType)
            }

        print("✅ Rules pre-parsed.")
        return rules

    def transform(self):
        """Orchestrates the entire transformation process."""
        self.load_and_parse_rules()
        rules = self.pre_parse_rules()

        print(f"📖 Reading CSV file: {self.csv_file}...")
        try:
            df = pd.read_csv(self.csv_file)
            print(f"✅ Successfully read {len(df)} rows.")
        except Exception as e:
            print(f"❌ Failed to read CSV file: {e}")
            return

        self.generate_skos_vocabularies(df)

        print("🔄 Start transforming data row by row...")
        for _, row in df.iterrows():
            self.process_row(row, rules)

        print("✅ Data transformation completed.")

    def save_output(self):
        """Saves the generated RDF graph to the output file."""
        print(f"💾 Saving RDF results to: {self.output_file}...")
        try:
            self.output_g.serialize(
                destination=self.output_file, format='xml')
            print(f"🎉 Success! RDF file generated.")
        except Exception as e:
            print(f"❌ Failed to save file: {e}")


def main():
    """Main function to run the transformer from the command line."""
    if len(sys.argv) != 4:
        print(
            "Usage: python transformer.py <rules_file.ttl> <csv_file.csv> <output_file.ttl>")
        sys.exit(1)

    rules_file, csv_file, output_file = sys.argv[1], sys.argv[2], sys.argv[3]

    transformer = LingoMapTransformer(rules_file, csv_file, output_file)
    transformer.transform()
    transformer.save_output()


if __name__ == "__main__":
    main()
