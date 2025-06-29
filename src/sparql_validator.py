"""
LingoMap Generated RDF Data SPARQL Validator
"""
# python sparql_validator.py fdic_banks_output.xml python3 -m venv lingomap_env

import sys
from rdflib import Graph
from typing import List, Dict

def run_sparql_validation(rdf_file: str):
    """
    Run a series of SPARQL validation queries on the specified RDF file.
    """
    print(f"🚀 Start SPARQL validation on generated RDF data...")
    print(f"📁 File: {rdf_file}")

    g = Graph()
    try:
        g.parse(rdf_file)
        print(f"✅ Successfully loaded {len(g)} triples.")
    except Exception as e:
        print(f"❌ Failed to load RDF file: {e}")
        return False

    # --- Define your validation check list ---
    # We mainly use ASK queries, which return True or False.
    # Our question is usually "Do we have any data that doesn't follow the rules?", so we expect the answer to be False.
    validation_checks: List[Dict] = [
        {
            "name": "Check: Are all branches (Branch) defined as geographic features (Feature)?",
            "query": """
                PREFIX fibo-be: <https://spec.edmcouncil.org/fibo/ontology/BE/LegalEntities/FormalBusinessOrganizations/>
                PREFIX geo: <http://www.opengis.net/ont/geosparql#>
                
                ASK WHERE {
                    ?branch a fibo-be:Branch .
                    FILTER NOT EXISTS { ?branch a geo:Feature . }
                }
            """,
            "expected_result": False, # Expect to find no such bad examples
            "error_message": "Found a branch entity that is not defined as geo:Feature."
        },
        {
            "name": "Check: Do all branches have addresses (hasAddress)?",
            "query": """
                PREFIX fibo-be: <https://spec.edmcouncil.org/fibo/ontology/BE/LegalEntities/FormalBusinessOrganizations/>
                PREFIX fibo-fnd-plc: <https://spec.edmcouncil.org/fibo/ontology/FND/Places/Addresses/>

                ASK WHERE {
                    ?branch a fibo-be:Branch .
                    FILTER NOT EXISTS { ?branch fibo-fnd-plc:hasAddress ?address . }
                }
            """,
            "expected_result": False,
            "error_message": "Found a branch entity that is missing an address association."
        },
        {
            "name": "Check: Do all geometries (Geometry) have WKT coordinates?",
            "query": """
                PREFIX sf: <http://www.opengis.net/ont/sf#>
                PREFIX geo: <http://www.opengis.net/ont/geosparql#>

                ASK WHERE {
                    ?geom a sf:Point .
                    FILTER NOT EXISTS { ?geom geo:asWKT ?wkt . }
                }
            """,
            "expected_result": False,
            "error_message": "Found a geometry entity that is missing a WKT coordinate."
        },
        {
            "name": "Check: Do all institutions (FinancialInstitution) have names?",
            "query": """
                PREFIX fibo-fbc: <https://spec.edmcouncil.org/fibo/ontology/FBC/FunctionalEntities/FinancialServicesEntities/>
                PREFIX lcc-lr: <https://www.omg.org/spec/LCC/Languages/LanguageRepresentation/>

                ASK WHERE {
                    ?inst a fibo-fbc:FinancialInstitution .
                    FILTER NOT EXISTS { ?inst lcc-lr:hasName ?name . }
                }
            """,
            "expected_result": False,
            "error_message": "Found a financial institution entity that is missing a name."
        },
    ]

    print("\n" + "="*50)
    print("🔍 Start executing validation queries...")
    print("="*50)

    all_passed = True
    for i, check in enumerate(validation_checks):
        print(f"\n({i+1}/{len(validation_checks)}) {check['name']}")
        
        try:
            query_result = g.query(check['query']).askAnswer
            
            if query_result == check['expected_result']:
                print("✅ PASS")
            else:
                print(f"❌ FAIL: {check['error_message']}")
                all_passed = False
        except Exception as e:
            print(f"❌ QUERY ERROR: Error executing query: {e}")
            all_passed = False

    print("\n" + "="*50)
    if all_passed:
        print("🎉 Congratulations! All SPARQL functional validations passed!")
    else:
        print("⚠️ Some validations failed, please check the generated RDF data.")
    print("="*50)

    return all_passed


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python sparql_validator.py <generated_rdf_file.ttl>")
        sys.exit(1)
    
    rdf_file = sys.argv[1]
    success = run_sparql_validation(rdf_file)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()