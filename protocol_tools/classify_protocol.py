import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ollama
from basic_tools.config import PROTOCOL_MODEL

classify_prompt = """
    You are an expert at classifying stem cell protocol queries. 
    First, classify the query into exactly one category:
    - Factual: Requests for specific protocol details
    - Analytical: Requests for comparisons/analysis
    - Contextual: Requests requiring protocol interpretation
    
    Then identify the extraction target from these options:
    - reagents: Chemical/components lists
    - timeline: Protocol stages/durations
    - efficiency: Yield metrics
    - critical_steps: Important protocol notes
    
    Return ONLY in this format: "Category|Target"
    Example: "Factual|reagents
    """

def classify_protocol(chunk):
    response = ollama.chat(
        model=PROTOCOL_MODEL,
        messages=[
            {"role": "system", "content": classify_prompt},
            {"role": "user", "content": chunk}
        ],
        options={"temperature": 0}
    )

    result = response['message']['content'].strip()
    try:
        category, target = result.split("|")
        return category.lower(), target.lower()
    except:
        # Fallback for malformed responses
        return ("factual", "reagents")
