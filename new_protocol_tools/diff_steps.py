import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.llms.ollama import Ollama 
from langchain.prompts import ChatPromptTemplate
from basic_tools.config import PROTOCOL_MODEL

def differentiation_steps(summary):  
    prompt_template = """
    You are analyzing summaries of a stem cell differentiation protocol.

    Given the following summary, identify if it refers to a specific stage in a cell differentiation process. If it does, extract the step in the following format:

    Step X: [Name of the differentiation stage]  
    Duration: [State the duration if available, else write 'Not specified']  
    [One paragraph describing the procedure in that stage]

    Guidelines:
    - Step 0 must describe the condition of the undifferentiated cells before any differentiation starts.
    - Subsequent steps (Step 1, Step 2...) should correspond to actual lineage commitment or specification stages (e.g., mesoderm induction, cardiac mesoderm, progenitor isolation, maturation).
    - Ignore steps that are purely technical (e.g., plating, passaging, media change for recordings) unless they are directly involved in differentiation.
    - If no differentiation-related action is described, return: "No differentiation step in this summary."

    Here is the summary:
    {summary}

    """

    prompt_template = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt_template.format(summary=summary)
    model = Ollama(model=PROTOCOL_MODEL)
    result = model.invoke(prompt)

    return result.strip()
