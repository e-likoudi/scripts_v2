import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ollama
from basic_tools.config import PROTOCOL_MODEL

def detail_protocol(summary, detail_level):
    prompt = f"""
    ACT AS A STEM CELL PROTOCOL ANALYST. Extract QUANTITATIVE DATA from:
    {summary}

    === EXTRACTION RULES ===
    Detail Level: {detail_level.upper()}
    1. Efficiency: "[Metric]: X% | [Method]"
    2. QC Metrics: "[Parameter] | [Value] | [Threshold]"
    3. Yield: "[Cell Type]: X% | [Day]"
    
    {"INCLUDE technical specifics" if detail_level == "high" else "Summarize key metrics"}
    """
    
    response = ollama.chat(
        model=PROTOCOL_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": summary}
        ],
        options={"temperature": 0}
    )
    
    raw = response['message']['content'].strip()
    return _parse_detail_data(raw)

def _parse_detail_data(raw):
    data = {"efficiency": [], "metrics": [], "yield": []}
    for line in raw.split('\n'):
        line = line.strip()
        if "%" in line:
            if "Yield" in line:
                data["yield"].append(line)
            else:
                data["efficiency"].append(line)
        elif "|" in line:
            data["metrics"].append(line)
    return data