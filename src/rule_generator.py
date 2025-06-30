import json
import logging
from google.generativeai.generative_models import GenerativeModel
import streamlit as st
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_conditional_rule_with_llm(
    column_name: str,
    column_profile: Dict[str, Any],
    entities: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Uses an LLM to analyze a column and determine if it should be a conditional rule.
    """
    logging.info(f"Analyzing column '{column_name}' for conditional rule generation.")
    model = GenerativeModel('gemini-1.5-flash')

    # Format the input data for the prompt
    formatted_profile = json.dumps(column_profile, indent=2)
    entity_list_str = ", ".join([f"`{e.get('entityId')}`" for e in entities if 'error' not in e])

    prompt = f"""
# ROLE
You are an expert ETL (Extract, Transform, Load) and Semantic Data Engineer, specializing in FIBO.

# TASK
Analyze the following column profile. Determine if it is a "flag" or "status" column that should be used to define a conditional transformation rule. Columns with few unique values (e.g., 0/1, Y/N, True/False) are excellent candidates.

If it IS a candidate for a rule, generate a JSON object describing the rule. If it is NOT, return a JSON object with the key `"isRule": false`.

# 1. COLUMN PROFILE
- Column Name: `{column_name}`
- Profile:
```json
{formatted_profile}

# 2. AVAILABLE ENTITIES
The rule must target one of these existing entities: {entity_list_str}

# Instructions
1. Analyze: Is this a flag, status, or type code?
2. If YES (it is a rule candidate):
  * Infer which entity the rule should apply to (e.g., a rule about a branch applies to BankBranchEntity).
  * Define the condition (e.g., when the column value equals "1").
  * Define the action (e.g., add a specific class like fibo-fnd-org-org:Headquarters).
  * Return a JSON object for the rule.
3. If NO (it's just regular data):
  * Return {{"isRule": false}}.
  
# OUTPUT FORMAT (for a conditional rule)
    {{
        "isRule": true,
        "ruleType": "ConditionalTypeAssignment",
        "ruleId": "rule_AssignHeadquartersType_For_MAINOFF",
        "sourceColumn": "{column_name}",
        "condition": {{
        "operator": "equals",
        "value": "1"
    }},
        "targetEntity": "BankBranchEntity",
        "action": {{
        "type": "addClass",
        "value": "fibo-fnd-org-org:Headquarters"
    }},
        "justification": "When MAINOFF is 1, the branch is the main office, which corresponds to the FIBO concept of a Headquarters."
    }}
"""

    try:
        logging.info(f"Sending conditional rule prompt for '{column_name}' to Gemini...")
        response = model.generate_content(prompt)
        cleaned_text = response.text.strip()
        # 2. 防禦性檢查：在解析前，先確認它至少看起來像一個 JSON 物件
        if cleaned_text.startswith('{') and cleaned_text.endswith('}'):
            rule_suggestion = json.loads(cleaned_text)
            return rule_suggestion
        else:
            # 3. 如果它不是 JSON，就記錄下來並安全地回傳
            logging.warning(f"LLM did not return a valid JSON for '{column_name}' rule check. Response: '{cleaned_text}'")
            return {"isRule": False, "justification": "LLM judged this is not a rule candidate."}
    except json.JSONDecodeError as e: # 更具體的錯誤捕捉
        logging.error(f"JSONDecodeError during conditional rule generation for '{column_name}': {e}. Response: '{response.text}'")
        return {"error": str(e), "isRule": False}
    except Exception as e:
        logging.error(f"An error occurred during conditional rule generation for '{column_name}': {e}")
        return {"error": str(e), "isRule": False}
