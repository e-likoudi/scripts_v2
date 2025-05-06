import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ollama
from basic_tools.config import PROTOCOL_MODEL

def analytical_protocol(chunks):
    analytical_prompt = """
    ACT AS A STEM CELL PROTOCOL PARSER. Extract EXACTLY as specified:

    === QUERY ===
    DATA TYPE: [Efficiency/QC/Metrics]
    FORMAT:
    - Efficiency: "Yield: X% | Method: [Technique]"
    - QC: "Marker: [Name] | Threshold: [Value]"
    - Metrics: "Parameter: [Name] | Value: [Data]"

    === RULES ===
    1. Write "Not specified" for missing data
    2. Include confidence intervals if present
    3. Prioritize table/bullet data

    === OUTPUT ===
    [Only the extracted data in specified format]
    """

    response = ollama.chat(
        model=PROTOCOL_MODEL,
        messages=[
            {"role": "system", "content": analytical_prompt},
            {"role": "user", "content": chunks}
        ],
        options={"temperature": 0}
    )
    raw = response['message']['content'].strip()
    lines = [line.strip() for line in raw.split('\n') if line.strip()]

    return {
            "efficiency": [l for l in lines if "Yield:" in l],
            "qc": [l for l in lines if "Marker:" in l],
            "metrics": [l for l in lines if "Parameter:" in l]
        }    