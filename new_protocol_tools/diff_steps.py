import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.llms.ollama import Ollama 
from langchain.prompts import ChatPromptTemplate
from basic_tools.config import PROTOCOL_MODEL

def differentiation_steps(summaries_list):  
    prompt_template = """
    Extract only the DIFFERENTIATION STEPS from the following protocol.

    Reproduce them in the same exact format as shown below:
    - Step number and title 
    - Duration
    - Procedure text in paragraph form

    Step 0 should be about the undiferentiated embryonic stem cells.
    Do NOT summarize, explain, paraphrase, or change anything.
    Keep the original order and structure. Return only the differentiation steps as they appear.

    Here is the full protocol:
    {documents}


    """
    formatted_summaries = "\n".join(summary for summary in summaries_list)

    prompt_template = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt_template.format(documents=formatted_summaries)

    model = Ollama(model=PROTOCOL_MODEL)
    response = model.invoke(prompt)

    return response