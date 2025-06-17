import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.llms.ollama import Ollama
from basic_tools.config import PROTOCOL_MODEL

def create_protocol(sorted_steps, cell_line):
    prompt = """
    You are an expert in biological protocols.
    You are given a series of steps in a differentiation protocol for stem cells.
    Your task is to refine these steps into a clear and concise protocol format.
    Take into account the following:
    - Cell line and target information: {cell_line}
    - Differentiation stages, duration, description and specific steps: {stages}

    Each step should be clearly numbered and formatted, with a focus on clarity and precision.
    The protocol should include the following:
    - Step 0: The condition of the undifferentiated cells before any differentiation starts.
    - Steps 1 through n: The differentiation process, including any specific signals, medium changes, or lineage-commitment steps.
    - The last step: The final differentiated state of the cells, including any markers or characteristics that define this state.
    
    """

    formatted_prompt = prompt.format(stages=sorted_steps, cell_line=cell_line)

    model = Ollama(model=PROTOCOL_MODEL)
    response = model.invoke(formatted_prompt)

    return response