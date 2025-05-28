import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ollama
from basic_tools.config import PROTOCOL_MODEL

def phase_protocol(summary, phase):
    phase_prompts = {
        "early": "Extract INITIAL SETUP details:",
        "mid": "Extract DIFFERENTIATION PROCESS details:",
        "late": "Extract MATURATION/ASSESSMENT details:"
    }
    
    prompt = f"""
    ACT AS A STEM CELL PROTOCOL SPECIALIST. {phase_prompts.get(phase, "")}
    Focus on TIMELINE and REAGENTS from this protocol summary:
    {summary}

    === EXTRACTION RULES ===
    1. Timeline: "Day X-Y: [Event] | Details: [Specifics]"
    2. Reagents: "[Name] | [Role] | [Concentration]"
    3. Media: "[Base Media] | [Additives]"
    
    Include ONLY data relevant to the current phase ({phase}).
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
    return _parse_phase_data(raw)

def _parse_phase_data(raw):
    data = {"timeline": [], "reagents": [], "media": []}
    for line in raw.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith("Day"):
            data["timeline"].append(line)
        elif "|" in line:
            if any(term in line.lower() for term in ["media", "dmem", "rpm"]):
                data["media"].append(line)
            else:
                data["reagents"].append(line)
    return data