import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.llms.ollama import Ollama 
from langchain.prompts import ChatPromptTemplate
from basic_tools.config import PROTOCOL_MODEL

class IdentifyDetails:
    @staticmethod
    
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
            "duration": "[Duration in hours or days, e.g., '24 hours', '3 days']"
        """
        durations = []
        
        for step in sorted_steps:
            prompt_template = ChatPromptTemplate.from_template(durations_prompt)
            prompt = prompt_template.format(step=step)
            model = Ollama(model=PROTOCOL_MODEL)
            result = model.invoke(prompt)

            durations.append(result)

        return durations
    
    def basic_media(documents):
        media_prompt = """
        You are an expert in stem cell biology analyzing protocol details. 
        For each part of the protocol you are given, identify the basic media used in it.

        Text:
        {text}

        Respond in the following format:
            "media": "[List of media used, e.g., 'mTeSR1, RPMI+B27+Activin A']"
        Or if no media is used:
            "media": "No media used"
        """
        
        media_results = []
        
        prompt_template = ChatPromptTemplate.from_template(media_prompt)
        prompt = prompt_template.format(text=documents)
        model = Ollama(model=PROTOCOL_MODEL)
        result = model.invoke(prompt)

        media_results.append(result)

        return media_results
    
    def serums_supplements(documents):
        serums_prompt = """
        You are an expert in stem cell biology analyzing protocol steps. 
        For each part of the protocol you are given, identify the serums and supplements used in that part.

        Text:
        {text}

        Respond in the following format:
            "serums_supplements": "[List of serums and supplements used, e.g., 'FBS, B27']"
        Or if no serums or supplements are used:
            "serums_supplements": "No serums or supplements used"
        """
        
        serums_results = []
        
        prompt_template = ChatPromptTemplate.from_template(serums_prompt)
        prompt = prompt_template.format(text=documents)
        model = Ollama(model=PROTOCOL_MODEL)
        result = model.invoke(prompt)

        serums_results.append(result)

        return serums_results
    
    def growth_factors(documents):
        growth_factors_prompt = """
        You are an expert in stem cell biology analyzing protocol steps. 
        For each part of the protocol you are given, identify the growth factors used in that part.
        
        Text:
        {text}

        Respond in the following format:
            "growth_factors": "[List of growth factors used, e.g., 'BMP4, FGF2']"
        Or if no growth factors are used:
            "growth_factors": "No growth factors used"
        """
        
        growth_factors_results = []
        
        prompt_template = ChatPromptTemplate.from_template(growth_factors_prompt)
        prompt = prompt_template.format(text=documents)
        model = Ollama(model=PROTOCOL_MODEL)
        result = model.invoke(prompt)

        growth_factors_results.append(result)

        return growth_factors_results
    
    def cytokines_supplements(documents):
        cytokines_prompt = """
        You are an expert in stem cell biology analyzing protocol steps. 
        For each part of the protocol you are given, identify the cytokines and chemical supplements used in that part.
    
        Text:
        {text}

        Respond in the following format:
            "cytokines_supplements": "[List of cytokines and supplements used, e.g., 'IL-6, TGF-beta']"
        Or if no cytokines or supplements are used:
            "cytokines_supplements": "No cytokines or supplements used"
        """
        
        cytokines_results = []
        
        prompt_template = ChatPromptTemplate.from_template(cytokines_prompt)
        prompt = prompt_template.format(text=documents)
        model = Ollama(model=PROTOCOL_MODEL)
        result = model.invoke(prompt)

        cytokines_results.append(result)

        return cytokines_results

    def passaging(documents):
        passaging_prompt = """
        You are an expert in stem cell biology analyzing protocol steps. 
        For each part of the protocol you are given, identify if it involves passaging of cells.
    
        Text:
        {text}

        Respond in the following format:
            "passaging": "[Yes/No]"
        """
        
        passaging_results = []
        
        prompt_template = ChatPromptTemplate.from_template(passaging_prompt)
        prompt = prompt_template.format(text=documents)
        model = Ollama(model=PROTOCOL_MODEL)
        result = model.invoke(prompt)

        passaging_results.append(result)

        return passaging_results

    def gene_markers(documents):
        gene_markers_prompt = """
        You are an expert in stem cell biology analyzing protocol steps. 
        For each part of the protocol you are given, identify the gene markers associated with that part.

        Text:
        {text}

        Respond in the following format:
            "gene_markers": "[List of gene markers, e.g., 'cTNT, SOX2']"
        Or if no gene markers are mentioned:
            "gene_markers": "No gene markers mentioned"
        """
        
        gene_markers_results = []
        
        prompt_template = ChatPromptTemplate.from_template(gene_markers_prompt)
        prompt = prompt_template.format(text=documents)
        model = Ollama(model=PROTOCOL_MODEL)
        result = model.invoke(prompt)

        gene_markers_results.append(result)

        return gene_markers_results
    


__all__ = ["calculate_durations",
           "basic_media",
           "serums_supplements",
           "growth_factors",
           "cytokines_supplements",
           "passaging",
           "gene_markers"]