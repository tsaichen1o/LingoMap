"""
LingoMap Turtle Mapping Validation Tool
Author: TsaiChen LO & AI Assistant
Date: 2025-06-27
"""

import sys
from rdflib import Graph, RDF, RDFS, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD, SKOS
from typing import Dict, List, Set, Tuple
import json
from datetime import datetime
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import compact_uri

class MappingValidator:
    """Turtle Mapping Validation Tool"""
    
    def __init__(self, mapping_file: str):
        self.mapping_file = mapping_file
        self.g = Graph()
        self.mymap = Namespace('http://example.com/mapping-schema#')
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'file': mapping_file,
            'checks': {},
            'statistics': {},
            'issues': [],
            'warnings': [],
            'errors': []
        }
    
    def load_mapping(self) -> bool:
        """Load mapping file"""
        try:
            self.g.parse(self.mapping_file, format='turtle')
            print(f"✅ Successfully loaded mapping file: {self.mapping_file}")
            return True
        except Exception as e:
            print(f"❌ Failed to load file: {e}")
            self.results['errors'].append(f"Failed to load file: {e}")
            return False

    def check_syntax(self) -> bool:
        """Check syntax correctness"""
        print("\n🔍 Checking syntax...")
        
        try:
            # Calculate triple count
            triple_count = len(self.g)
            self.results['statistics']['triple_count'] = triple_count
            
            # Check basic syntax
            syntax_valid = True
            issues = []
            
            # Check if there are valid triples
            if triple_count == 0:
                issues.append("File contains no valid triples")
                syntax_valid = False
            
            # Check namespaces
            namespaces = list(self.g.namespace_manager.namespaces())
            self.results['statistics']['namespace_count'] = len(namespaces)
            
            if len(namespaces) < 5:  # At least there should be basic namespaces
                issues.append("Too few namespaces")
                syntax_valid = False
            
            if syntax_valid:
                print(f"✅ Syntax check passed - {triple_count} triples, {len(namespaces)} namespaces")
            else:
                print(f"❌ Syntax check failed")
                for issue in issues:
                    print(f"   - {issue}")
            
            self.results['checks']['syntax'] = syntax_valid
            self.results['issues'].extend(issues)
            return syntax_valid
            
        except Exception as e:
            print(f"❌ Syntax check error: {e}")
            self.results['errors'].append(f"Syntax check error: {e}")
            return False
    
    def check_structure(self) -> bool:
        """Check structure completeness"""
        print("\n📋 Checking structure completeness...")
        
        try:
            # Check mapping types
            mapping_types = {}
            for s, p, o in self.g.triples((None, RDF.type, None)):
                type_uri = str(o)
                if type_uri not in mapping_types:
                    mapping_types[type_uri] = []
                mapping_types[type_uri].append(s)
            
            self.results['statistics']['mapping_types'] = {
                k.split('#')[-1] if '#' in k else k.split('/')[-1]: len(v) 
                for k, v in mapping_types.items()
            }
            
            # Check required mapping types
            required_types = [
                'EntityMapping',
                'AssociationMapping', 
                'ColumnMapping',
                'IdentifierMapping'
            ]
            
            missing_types = []
            for req_type in required_types:
                found = False
                for type_uri in mapping_types.keys():
                    if req_type in type_uri:
                        found = True
                        break
                if not found:
                    missing_types.append(req_type)
            
            structure_valid = len(missing_types) == 0
            
            if structure_valid:
                print("✅ Structure check passed")
                for type_name, count in self.results['statistics']['mapping_types'].items():
                    print(f"   - {type_name}: {count} instances")
            else:
                print(f"❌ Structure check failed - missing required types: {missing_types}")
            
            self.results['checks']['structure'] = structure_valid
            if missing_types:
                self.results['issues'].extend([f"Missing required type: {t}" for t in missing_types])
            
            return structure_valid
            
        except Exception as e:
            print(f"❌ Structure check error: {e}")
            self.results['errors'].append(f"Structure check error: {e}")
            return False
    
    def check_entities(self) -> bool:
        """Check entity mappings"""
        print("\n🏢 Checking entity mappings...")
        
        try:
            # Find all entity mappings
            entities = []
            for s, p, o in self.g.triples((None, RDF.type, None)):
                if 'EntityMapping' in str(o):
                    label = self.g.value(s, RDFS.label)
                    comment = self.g.value(s, RDFS.comment)
                    entities.append({
                        'uri': str(s),
                        'label': str(label) if label else 'No label',
                        'comment': str(comment) if comment else 'No description'
                    })
            
            self.results['statistics']['entities'] = entities
            
            # Check entity associations
            linked_entities = set()
            for s, p, o in self.g.triples((None, self.mymap.linksEntity, None)):
                linked_entities.add(str(o))
            for s, p, o in self.g.triples((None, self.mymap.toEntity, None)):
                linked_entities.add(str(o))
            
            entity_uris = {e['uri'] for e in entities}
            isolated_entities = entity_uris - linked_entities
            
            if isolated_entities:
                print(f"⚠️ Found {len(isolated_entities)} isolated entities:")
                for uri in isolated_entities:
                    label = next((e['label'] for e in entities if e['uri'] == uri), 'Unknown')
                    print(f"   - {label} ({uri})")
                    self.results['warnings'].append(f"Isolated entity: {label}")
            else:
                print("✅ All entities are linked")
            
            print(f"📊 Entity statistics: {len(entities)} entity mappings")
            for entity in entities:
                print(f"   - {entity['label']}: {entity['comment']}")
            
            self.results['checks']['entities'] = len(entities) > 0
            return len(entities) > 0
            
        except Exception as e:
            print(f"❌ Entity check error: {e}")
            self.results['errors'].append(f"Entity check error: {e}")
            return False
    
    def check_associations(self) -> bool:
        """Check association mappings"""
        print("\n🔗 Checking association mappings...")
        
        try:
            associations = []
            for s, p, o in self.g.triples((None, RDF.type, None)):
                if 'AssociationMapping' in str(o):
                    links_entity = self.g.value(s, self.mymap.linksEntity)
                    to_entity = self.g.value(s, self.mymap.toEntity)
                    using_property = self.g.value(s, self.mymap.usingProperty)
                    
                    associations.append({
                        'uri': str(s),
                        'from': str(links_entity) if links_entity else 'Unknown',
                        'to': str(to_entity) if to_entity else 'Unknown',
                        'property': str(using_property) if using_property else 'Unknown'
                    })
            
            self.results['statistics']['associations'] = associations
            
            print(f"📊 Association statistics: {len(associations)} association mappings")
            for assoc in associations:
                from_label = assoc['from'].split('#')[-1] if '#' in assoc['from'] else assoc['from'].split('/')[-1]
                to_label = assoc['to'].split('#')[-1] if '#' in assoc['to'] else assoc['to'].split('/')[-1]
                print(f"   - {from_label} → {to_label}")
            
            self.results['checks']['associations'] = len(associations) > 0
            return len(associations) > 0
            
        except Exception as e:
            print(f"❌ Association check error: {e}")
            self.results['errors'].append(f"Association check error: {e}")
            return False
    
    def check_columns(self) -> bool:
        """Check column mappings"""
        print("\n📊 Checking column mappings...")
        
        try:
            columns = []
            for s, p, o in self.g.triples((None, RDF.type, None)):
                if 'ColumnMapping' in str(o):
                    source_column = self.g.value(s, self.mymap.sourceColumn)
                    maps_to_property = self.g.value(s, self.mymap.mapsToProperty)
                    part_of = self.g.value(s, self.mymap.partOf)
                    data_type = self.g.value(s, self.mymap.hasDataType)
                    
                    columns.append({
                        'uri': str(s),
                        'source_column': str(source_column) if source_column else 'Unknown',
                        'maps_to_property': str(maps_to_property) if maps_to_property else 'Unknown',
                        'part_of': str(part_of) if part_of else 'Unknown',
                        'data_type': str(data_type) if data_type else 'Unknown'
                    })
            
            self.results['statistics']['columns'] = columns
            
            # Check if columns have entity associations
            columns_with_entity = len([c for c in columns if c['part_of'] != 'Unknown'])
            
            print(f"📊 Column statistics: {len(columns)} column mappings")
            print(f"   - With entity associations: {columns_with_entity} columns")
            print(f"   - Without entity associations: {len(columns) - columns_with_entity} columns")
            
            if len(columns) - columns_with_entity > 0:
                print("⚠️ Found columns without entity associations:")
                for col in columns:
                    if col['part_of'] == 'Unknown':
                        print(f"   - {col['source_column']}")
                        self.results['warnings'].append(f"Column without entity association: {col['source_column']}")
            
            self.results['checks']['columns'] = len(columns) > 0
            return len(columns) > 0
            
        except Exception as e:
            print(f"❌ Column check error: {e}")
            self.results['errors'].append(f"Column check error: {e}")
            return False
    
    def check_identifiers(self) -> bool:
        """Check identifier mappings"""
        print("\n🆔 Checking identifier mappings...")
        
        try:
            identifiers = []
            for s, p, o in self.g.triples((None, RDF.type, None)):
                if 'IdentifierMapping' in str(o):
                    source_column = self.g.value(s, self.mymap.sourceColumn)
                    identifies_entity = self.g.value(s, self.mymap.identifiesEntity)
                    identifier_scheme = self.g.value(s, self.mymap.identifierScheme)
                    
                    identifiers.append({
                        'uri': str(s),
                        'source_column': str(source_column) if source_column else 'Unknown',
                        'identifies_entity': str(identifies_entity) if identifies_entity else 'Unknown',
                        'scheme': str(identifier_scheme) if identifier_scheme else 'Unknown'
                    })
            
            self.results['statistics']['identifiers'] = identifiers
            
            print(f"📊 Identifier statistics: {len(identifiers)} identifier mappings")
            for ident in identifiers:
                print(f"   - {ident['source_column']} → {ident['identifies_entity'].split('#')[-1]}")
            
            self.results['checks']['identifiers'] = len(identifiers) > 0
            return len(identifiers) > 0
            
        except Exception as e:
            print(f"❌ Identifier check error: {e}")
            self.results['errors'].append(f"Identifier check error: {e}")
            return False
    
    def check_namespaces(self) -> bool:
        """Check namespaces"""
        print("\n🔗 Checking namespaces...")
        
        try:
            namespaces = list(self.g.namespace_manager.namespaces())
            
            # Check required namespaces
            required_ns = ['rdf', 'rdfs', 'xsd', 'mymap']
            missing_ns = []
            
            ns_prefixes = [ns[0] for ns in namespaces]
            for req in required_ns:
                if req not in ns_prefixes:
                    missing_ns.append(req)
            
            if missing_ns:
                print(f"❌ Missing required namespaces: {missing_ns}")
                self.results['issues'].extend([f"Missing namespace: {ns}" for ns in missing_ns])
            else:
                print("✅ All required namespaces are defined")
            
            print(f"📊 Namespace statistics: {len(namespaces)} namespaces")
            
            # Show important namespaces
            important_ns = ['rdf', 'rdfs', 'xsd', 'skos', 'mymap', 'fibo', 'geo', 'schema']
            for prefix, uri in namespaces:
                if prefix in important_ns:
                    print(f"   - {prefix}: {uri}")
            
            self.results['statistics']['namespaces'] = {
                'total': len(namespaces),
                'list': [(prefix, str(uri)) for prefix, uri in namespaces]
            }
            
            self.results['checks']['namespaces'] = len(missing_ns) == 0
            return len(missing_ns) == 0
            
        except Exception as e:
            print(f"❌ Namespace check error: {e}")
            self.results['errors'].append(f"Namespace check error: {e}")
            return False
    
    def check_concepts(self) -> bool:
        """Check concept generation mappings"""
        print("\n🏷️ Checking concept generation mappings...")
        
        try:
            concepts = []
            for s, p, o in self.g.triples((None, RDF.type, None)):
                if 'ConceptGenerationMapping' in str(o):
                    label_column = self.g.value(s, self.mymap.labelColumn)
                    notation_column = self.g.value(s, self.mymap.notationColumn)
                    concept_id_prefix = self.g.value(s, self.mymap.conceptIdPrefix)
                    scheme_label = self.g.value(s, self.mymap.schemeLabel)
                    
                    concepts.append({
                        'uri': str(s),
                        'label_column': str(label_column) if label_column else 'Unknown',
                        'notation_column': str(notation_column) if notation_column else 'Unknown',
                        'concept_id_prefix': str(concept_id_prefix) if concept_id_prefix else 'Unknown',
                        'scheme_label': str(scheme_label) if scheme_label else 'Unknown'
                    })
            
            self.results['statistics']['concepts'] = concepts
            
            print(f"📊 Concept generation statistics: {len(concepts)} concept generation mappings")
            for concept in concepts:
                print(f"   - {concept['scheme_label']}: {concept['label_column']} → {concept['concept_id_prefix']}")
            
            self.results['checks']['concepts'] = len(concepts) >= 0  # Concept mappings are optional
            return True
            
        except Exception as e:
            print(f"❌ Concept check error: {e}")
            self.results['errors'].append(f"Concept check error: {e}")
            return False

    # ==================================================================
    #  Check if external vocabulary terms actually exist online
    # ==================================================================
    def check_vocabulary_existence(self) -> bool:
        """Check if external vocabulary terms actually exist online."""
        print("\n🌍 Checking vocabulary existence online (this may take a moment)...")
        
        # Collect all external URIs to check
        uris_to_check = set()
        properties_to_check = [
            self.mymap.mapsToClass,
            self.mymap.usingProperty,
            self.mymap.mapsToProperty,
            self.mymap.identifierScheme
        ]
        
        for prop in properties_to_check:
            for s, p, o in self.g.triples((None, prop, None)):
                if isinstance(o, URIRef) and "example.com" not in str(o):
                    uris_to_check.add(str(o))
        
        print(f"   - Found {len(uris_to_check)} unique external URIs to check.")
        
        if not uris_to_check:
            print("   - No external URIs to check.")
            self.results['checks']['vocabulary_existence'] = True
            return True

        invalid_uris = []
        
        # Use thread pool to parallelize requests for faster processing
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_uri = {executor.submit(self._check_uri, uri): uri for uri in uris_to_check}
            for i, future in enumerate(as_completed(future_to_uri)):
                uri = future_to_uri[future]
                try:
                    is_valid = future.result()
                    if not is_valid:
                        invalid_uris.append(uri)
                    # Show progress
                    sys.stdout.write(f"\r   - Progress: {i+1}/{len(uris_to_check)} URIs checked...")
                    sys.stdout.flush()
                except Exception as exc:
                    invalid_uris.append(uri)
                    self.results['errors'].append(f"Error checking URI {uri}: {exc}")

        print("\n   - Check completed.")

        if not invalid_uris:
            print("✅ All external vocabulary URIs are valid and reachable.")
            self.results['checks']['vocabulary_existence'] = True
            return True
        else:
            print(f"❌ Found {len(invalid_uris)} invalid or unreachable URIs:")
            for uri in invalid_uris:
                print(f"   - {uri}")
                self.results['errors'].append(f"Invalid or unreachable URI: {uri}")
            self.results['checks']['vocabulary_existence'] = False
            return False

    def _check_uri(self, uri: str) -> bool:
        """Helper function to check a single URI."""
        
        # Whitelist: Known issues but actually usable URIs
        whitelist = [
            "https://www.omg.org/spec/Commons/Locations/hasLongitude",
            "https://www.omg.org/spec/Commons/Locations/hasLatitude", 
            "https://www.omg.org/spec/LCC/Languages/LanguageRepresentation/hasName",
            "https://www.omg.org/spec/Commons/Identifiers/isIdentifiedBy",
            "https://www.omg.org/spec/Commons/Classifiers/isClassifiedBy",
        ]
        
        if uri in whitelist:
            return True
        
        try:
            # Use HEAD request, more efficient because we only need the status code
            # Add timeout and User-Agent to avoid being blocked
            headers = {'User-Agent': 'LingoMap-Validator/1.0'}
            response = requests.head(uri, allow_redirects=True, timeout=10, headers=headers)
            # 2xx status code means success
            if 200 <= response.status_code < 300:
                return True
            else:
                return False
        except requests.exceptions.RequestException:
            # Any network level error is considered unreachable
            return False
    # ==================================================================

    def generate_summary(self):
        """Generate validation summary"""
        print("\n" + "="*60)
        print("📋 Validation summary")
        print("="*60)
        
        # Calculate number of passed checks
        passed_checks = sum(1 for check in self.results['checks'].values() if check)
        total_checks = len(self.results['checks'])
        
        print(f"✅ Passed checks: {passed_checks}/{total_checks}")
        print(f"⚠️ Warning count: {len(self.results['warnings'])}")
        print(f"❌ Error count: {len(self.results['errors'])}")
        
        # Show check results
        for check_name, passed in sorted(self.results['checks'].items()):
            status = "✅" if passed else "❌"
            print(f"{status} {check_name}")
        
        # Show warnings
        if self.results['warnings']:
            print(f"\n⚠️ Warnings:")
            for warning in self.results['warnings']:
                print(f"   - {warning}")
        
        # Show errors
        if self.results['errors']:
            print(f"\n❌ Errors:")
            for error in self.results['errors']:
                print(f"   - {error}")
        
        # Overall assessment
        if len(self.results['errors']) == 0 and passed_checks == total_checks:
            print(f"\n🎉 Validation result: All checks passed")
        elif len(self.results['errors']) == 0:
            print(f"\n✅ Validation result: Basic pass (with warnings)")
        else:
            print(f"\n❌ Validation result: Needs repair")
    
    def save_report(self, output_file: str = None):
        """Save validation report"""
        if output_file is None:
            output_file = f"mapping_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Validation report saved: {output_file}")
        except Exception as e:
            print(f"❌ Failed to save report: {e}")
    
    def run_all_checks(self) -> bool:
        """Run all checks"""
        print("🚀 Starting validation of Turtle Mapping file...")
        print(f"📁 File: {self.mapping_file}")
        
        # Load file
        if not self.load_mapping():
            return False
        
        # Run all checks
        checks = [
            self.check_syntax,
            self.check_structure,
            self.check_entities,
            self.check_associations,
            self.check_columns,
            self.check_identifiers,
            self.check_namespaces,
            self.check_concepts,
            self.check_vocabulary_existence
        ]
        
        all_passed = True
        for check in checks:
            if not check():
                all_passed = False
        
        # Generate summary
        self.generate_summary()
        
        return all_passed

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python validate_mapping.py <mapping_file.ttl>")
        sys.exit(1)
    
    mapping_file = sys.argv[1]
    validator = MappingValidator(mapping_file)
    
    # Run validation
    success = validator.run_all_checks()
    
    # Save report
    validator.save_report()
    
    # Return exit code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()