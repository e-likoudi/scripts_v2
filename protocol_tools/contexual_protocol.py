import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ollama
from basic_tools.config import PROTOCOL_MODEL

def contextual_protocol(chunks):
    contextual_prompt = """
    ACT AS A STEM CELL PROTOCOL PARSER. Extract EXACTLY as specified:

    === QUERY ===
    DATA TYPE: [Warnings/Tips/References]
    FORMAT:
    - Warnings: "Risk: [Description] | Severity: [Level]"
    - Tips: "Tip: [Content] | Applicability: [Scope]"
    - References: "Source: [Citation] | Relevance: [Explanation]"

    === RULES ===
    1. Write "Not specified" for missing data
    2. Preserve urgency markers (!, CRITICAL)
    3. Capture footnotes/superscripts

    === OUTPUT ===
    [Only the extracted data in specified format]
    """

    response = ollama.chat(
        model=PROTOCOL_MODEL,
        messages=[
            {"role": "system", "content": contextual_prompt},
            {"role": "user", "content": chunks}
        ],
        options={"temperature": 0}
    )
    raw = response['message']['content'].strip()
    lines = [line.strip() for line in raw.split('\n') if line.strip()]
    
    return {
            "warnings": [l for l in lines if "Risk:" in l],
            "tips": [l for l in lines if "Tip:" in l],
            "references": [l for l in lines if "Source:" in l]
        }