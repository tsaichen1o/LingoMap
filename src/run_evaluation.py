import json
from rdflib import Graph, RDF, Namespace
import sys
import os

# Ensure that modules in the src folder can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from core_engine import CoreMappingEngine
except ImportError as e:
    print(f"❌ Error: Cannot import CoreMappingEngine. Please check your file structure. Error message: {e}")
    sys.exit(1)

class GoldenStandardParser:
    """
    A class specifically designed to parse your lingomap_rules.ttl golden standard file.
    """
    def __init__(self, rules_file_path):
        self.g = Graph()
        self.g.parse(rules_file_path, format='turtle')
        self.mymap = Namespace('http://example.com/mapping-schema#')
        self.golden_data = {}

    def parse(self):
        """Parse the file and convert it into a dictionary with sourceColumn as the key."""
        # Process ColumnMapping
        for s in self.g.subjects(RDF.type, self.mymap.ColumnMapping):
            col = str(self.g.value(s, self.mymap.sourceColumn))
            self.golden_data[col] = {
                'mapping_type': 'ColumnMapping',
                'maps_to_property': str(self.g.value(s, self.mymap.mapsToProperty))
            }

        # Process IdentifierMapping
        for s in self.g.subjects(RDF.type, self.mymap.IdentifierMapping):
            col = str(self.g.value(s, self.mymap.sourceColumn))
            self.golden_data[col] = {
                'mapping_type': 'IdentifierMapping',
                'identifies_entity': str(self.g.value(s, self.mymap.identifiesEntity))
            }
        return self.golden_data

def evaluate_engine():
    """
    Main evaluation function, coordinating the entire evaluation process.
    """
    print("🚀 Starting automated evaluation of LingoMap v2.0 engine...")
    print("=" * 50)

    # 1. Load the golden standard
    print("1. Parsing golden standard file `lingomap_rules.ttl`...")
    try:
        parser = GoldenStandardParser('lingomap_rules.ttl')
        golden_standard = parser.parse()
        print(f"   ✅ Successfully parsed {len(golden_standard)} golden standard mapping rules.")
    except Exception as e:
        print(f"   ❌ Failed to parse golden standard: {e}")
        return

    # 2. Initialize the AI core engine
    print("\n2. Initializing LingoMap AI core engine...")
    try:
        # Here we assume the dataset exists in a fixed path
        # In actual applications, you may need to make it more flexible
        import pandas as pd
        csv_path = "FDIC_Insured_Banks.csv"
        df = pd.read_csv(csv_path, low_memory=False)
        engine = CoreMappingEngine()
        print("   ✅ AI engine initialized successfully.")
    except Exception as e:
        print(f"   ❌ Failed to initialize AI engine or read data: {e}")
        return

    # 3. Execute and compare one by one
    print("\n3. Starting to compare one by one...")
    correct_predictions = 0
    total_predictions = 0
    results_log = []

    columns_to_test = list(golden_standard.keys())

    for i, column_name in enumerate(columns_to_test):
        print(f"\n--- ({i+1}/{len(columns_to_test)}) Testing field: `{column_name}` ---")
        
        # Get AI suggestion
        ai_suggestion = engine.suggest_mapping(column_name, df[column_name])
        
        # Get golden standard answer
        golden_answer = golden_standard[column_name]
        
        is_correct = False
        ai_key_property = ai_suggestion.get('recommended_property_uri') # Adjusted based on your `core_engine.py`

        # Comparison logic
        if ai_suggestion.get('mapping_type') == golden_answer['mapping_type']:
            if golden_answer['mapping_type'] == 'ColumnMapping' and ai_key_property == golden_answer['maps_to_property']:
                is_correct = True
            elif golden_answer['mapping_type'] == 'IdentifierMapping' and ai_suggestion.get('identifies_entity') == golden_answer['identifies_entity']:
                 is_correct = True # Simplified version: only compare type and target entity
        
        total_predictions += 1
        if is_correct:
            correct_predictions += 1
            result = "✅ PASS"
            print(f"   {result}: AI suggestion matches golden standard.")
        else:
            result = "❌ FAIL"
            print(f"   {result}: AI suggestion does not match golden standard.")
            print(f"     - Golden standard: {golden_answer}")
            print(f"     - AI suggestion: {ai_suggestion}")

        results_log.append({'column': column_name, 'result': result})

    # 4. Generate final report
    print("\n" + "=" * 50)
    print("📊 Evaluation summary report")
    print("=" * 50)

    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    
    print(f"Total number of tested fields: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Incorrect predictions: {total_predictions - correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    print("\nDetailed log:")
    for log in results_log:
        print(f"  - {log['column']}: {log['result']}")

if __name__ == "__main__":
    evaluate_engine()