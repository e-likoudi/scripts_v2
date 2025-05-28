import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ollama
from basic_tools.config import PROTOCOL_MODEL

def focus_protocol(summary, focus_areas):
    prompt = f"""
    ACT AS A STEM CELL PROTOCOL REVIEWER. Extract CRITICAL INFORMATION from:
    {summary}

    === FOCUS AREAS ===
    {", ".join(focus_areas)}

    === EXTRACTION RULES ===
    1. Warnings: "![Priority] [Description] | [Reason]"
    2. Tips: "[Tip] | [Application]"
    3. Notes: "NOTE: [Observation] | [Impact]"
    
    Prioritize items marked as CRITICAL/IMPORTANT.
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
    return _parse_focus_data(raw)

def _parse_focus_data(raw):
    data = {"warnings": [], "tips": [], "notes": []}
    for line in raw.split('\n'):
        line = line.strip()
        if "!" in line:
            data["warnings"].append(line)
        elif "★" in line:
            data["tips"].append(line)
        elif "NOTE:" in line:
            data["notes"].append(line)
    return data