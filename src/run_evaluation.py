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
                'mappingType': 'ColumnMapping', # 鍵名統一為 camelCase
                'mapsToProperty': str(self.g.value(s, self.mymap.mapsToProperty))
            }

        # Process IdentifierMapping
        for s in self.g.subjects(RDF.type, self.mymap.IdentifierMapping):
            col = str(self.g.value(s, self.mymap.sourceColumn))
            self.golden_data[col] = {
                'mappingType': 'IdentifierMapping', # 鍵名統一為 camelCase
                'mapsToProperty': str(self.g.value(s, self.mymap.identifierScheme)) # 在黃金準則中，我們比對的是 identifierScheme
            }
        
        # (您可以在此處加入對 ClassificationMapping 等其他類型的解析)
        # Process ClassificationMapping

        return self.golden_data

    def get_data(self):
        """Return the parsed golden data."""
        return self.golden_data


class Evaluator:
    """Compares AI suggestions against a golden standard."""

    def __init__(self, golden_standard_path: str):
        """
        Initializes the evaluator by parsing the golden standard file.
        """
        print(f"Loading golden standard from: {golden_standard_path}")
        if not os.path.exists(golden_standard_path):
            raise FileNotFoundError(f"Golden standard file not found at: {golden_standard_path}")
        
        parser = GoldenStandardParser(golden_standard_path)
        parser.parse()
        self.golden_data = parser.get_data()
        print("✅ Golden standard parsed successfully.")

    def evaluate(self, ai_suggestions: dict) -> dict:
        """
        Evaluates the AI suggestions against the loaded golden standard.
        """
        print("🚀 Starting evaluation...")
        correct_predictions = 0
        total_tested = 0
        results_log = []

        # 只對黃金準則中定義的欄位進行評估
        for column_name, golden_answer in self.golden_data.items():
            ai_suggestion = ai_suggestions.get(column_name)
            
            if not ai_suggestion or "error" in ai_suggestion:
                result = "❌ NOT_FOUND"
                is_correct = False
                ai_property = "N/A (AI did not provide a valid suggestion)"
            else:
                total_tested += 1
                is_correct = False
                
                # --- 這就是關鍵的修正 ---
                # 獲取 AI 和黃金準則的映射類型
                ai_type = ai_suggestion.get('mappingType')
                golden_type = golden_answer.get('mappingType')
                ai_property = ai_suggestion.get('mapsToProperty') # 統一獲取比較的屬性
                golden_property = golden_answer.get('mapsToProperty')

                # 1. 首先，檢查映射類型是否匹配
                if ai_type == golden_type:
                    # 2. 如果類型匹配，再檢查屬性是否匹配
                    if ai_property == golden_property:
                        is_correct = True
                
                result = "✅ PASS" if is_correct else "❌ FAIL"
            
            if is_correct:
                correct_predictions += 1
            
            # 在日誌中加入更豐富的資訊
            results_log.append({
                'column': column_name, 
                'result': result,
                'golden_type': golden_answer.get('mappingType'),
                'ai_type': ai_suggestion.get('mappingType', 'N/A') if ai_suggestion else 'N/A',
                'golden_property': golden_answer.get('mapsToProperty'),
                'ai_property': ai_property
            })
        
        accuracy = (correct_predictions / total_tested) * 100 if total_tested > 0 else 0
        print(f"✅ Evaluation complete. Accuracy: {accuracy:.2f}%")
        
        return {
            "total_tested": total_tested,
            "correct": correct_predictions,
            "incorrect": total_tested - correct_predictions,
            "accuracy": accuracy,
            "log": results_log
        }