from langchain_community.llms.ollama import Ollama 
from langchain.prompts import ChatPromptTemplate
from basic_tools.config import PROTOCOL_MODEL

def calculate_durations(sorted_steps):
    
    durations_prompt = """
    You are an expert in stem cell biology analyzing protocol steps. 
    For each step of the protocol you are given, determine the duration required for that step based on the provided information.

    Their format is: 
        "stage": "[Undifferentiated cells/Differentiation Process/Differentiated cells/No differentiation step]",  
        "reason": "One paragraph describing the procedure in that stage",  
        "specific_step": "[Optional: lineage specification if applicable]" 

    Step:
    {step}

    DO NOT make any changes to the "stage", "reason", or "specific_step" fields.
    Respond in the following format:
        "stage"
        "duration": "[Duration in hours or days, e.g., '24 hours', '3 days']"
        "reason"
        "specific_step"

    """
    durations = []
    
    for step in sorted_steps:
        prompt_template = ChatPromptTemplate.from_template(durations_prompt)
        prompt = prompt_template.format(step=step)
        model = Ollama(model=PROTOCOL_MODEL)
        result = model.invoke(prompt)

        durations.append(result)

    return durations