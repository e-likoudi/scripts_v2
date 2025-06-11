import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.llms.ollama import Ollama
from basic_tools.config import PROTOCOL_MODEL

def format_protocol_steps(protocol_steps):
    if not protocol_steps:
        return "No protocol steps available"
    
    output_lines = []
    for step in protocol_steps:
        output_lines.append(f"Step {step['step_num']}: {step['name']}")
        output_lines.append(f"Description: {step['description']}")
        output_lines.append("")  
    
    return "\n".join(output_lines).strip()

def create_protocol(merged_stages):
    prompt = """
    You are an expert in biological protocols.
    You are given the stages of a differentiation protocol, a description and the step they refer to.
    Refine the information to create a clear and concise protocol.

    Rules:
    - Step 0 must describe the condition of the undifferentiated cells before any differentiation starts.
    - Steps 1 through n must describe the differentiation process, including any specific signals, medium changes, or lineage-commitment steps.
    - Last step must describe the final differentiated state of the cells, including any markers or characteristics that define this state.
    - Each step should be clearly numbered and formatted.

    Input stages:
    {stages}
    """

    formatted_prompt = prompt.format(stages=merged_stages)

    model = Ollama(model=PROTOCOL_MODEL)
    response = model.invoke(formatted_prompt)

    return response