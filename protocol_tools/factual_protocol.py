import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ollama
from basic_tools.config import PROTOCOL_MODEL

def factual_protocol(chunks):
        
    # Use LLM to enhance the query for better precision
    factual_prompt = """
        ACT AS A STEM CELL PROTOCOL PARSER. Extract the following information EXACTLY as specified below:  

        === QUERY ===  
        DATA TYPE: [Reagents/Timings/Steps]  
        FORMAT:  
        - Reagents: "Name | Concentration | Supplier"  
        - Timings: "Day X: [Duration] - [Step]"  
        - Steps: "Step X: [Action] | Notes: [Details]"  

        === RULES ===  
        1. If data is missing, write "Not specified".  
        2. Do not ignore vague terms like "briefly" or "approximately".  
        3. Prioritize data from tables or bulleted lists.  

        === OUTPUT ===  
        [Only include the extracted data in the specified format.]  
    """

    response = ollama.chat(
            model=PROTOCOL_MODEL,
            messages=[
                {"role": "system", "content": factual_prompt},
                {"role": "user", "content": chunks}
            ],
            options={"temperature": 0}
        )
    raw = response['message']['content'].strip()

    lines = [line.strip() for line in raw.split('\n') if line.strip()]
    data = {"reagents": [], "timings": [], "steps": []}
        
    for line in lines:
        if "|" in line and "Not specified" not in line:
            if "Day" in line:
                data["timings"].append(line)
            elif "Step" in line:
                data["steps"].append(line)
            else:
                data["reagents"].append(line)
    return data