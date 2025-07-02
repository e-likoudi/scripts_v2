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
            "specific_step": "[Optional: lineage specification if applicable]",  
            "source_documents": "[List of source documents]"

        Step:
        {step}

        DO NOT make any changes to the "stage", "reason", "specific_step", or "source_documents" fields.
        Respond in the following format:
            "duration": "[Duration in hours or days, e.g., '24 hours', '3 days']"
        """
        updated_steps = []
    
        for step in sorted_steps:
            updated_step = step.copy()
            
            prompt_template = ChatPromptTemplate.from_template(durations_prompt)
            prompt = prompt_template.format(step=step)
            model = Ollama(model=PROTOCOL_MODEL)  
            result = model.invoke(prompt)
            
            duration = "Not specified"
            for line in result.splitlines():
                line = line.strip().strip('"').strip(',')
                if line.startswith('"duration":'):
                    duration = line.split(':', 1)[1].strip().strip('"')
                    break
                    
            updated_step["duration"] = duration
            updated_steps.append(updated_step)
        
        return updated_steps
    
    def basic_media(duration_steps):
        media_prompt = """
        You are an expert in stem cell biology analyzing protocol steps. 
        Analyze the following protocol document to identify the basic media used in this step.
        Focus only on identifying media components mentioned in the text.

        Protocol Document:
        {document_text}

        Respond in the following format:
            "media": "[List of media used, e.g., 'mTeSR1, RPMI+B27+Activin A']"
        Or if no media is used:
            "media": "No media used"
        """
                
        updated_steps = []
    
        for step in duration_steps:
            updated_step = step.copy()
            
            document_text = step.get('source_documents', '')
            prompt_template = ChatPromptTemplate.from_template(media_prompt)
            prompt = prompt_template.format(document_text=document_text)
            model = Ollama(model=PROTOCOL_MODEL)  
            result = model.invoke(prompt)
            
            media = "No media used"
            for line in result.splitlines():
                line = line.strip().strip('"').strip(',')
                if line.startswith('"media":'):
                    media = line.split(':', 1)[1].strip().strip('"')
                    break
                    
            updated_step["basic_media"] = media
            updated_steps.append(updated_step)
        
        return updated_steps
    
    def serums_supplements(bm_steps):
        serums_prompt = """
        You are an expert in stem cell biology analyzing protocol documents. 
        Analyze the following protocol document to identify serums and supplements used in this step.
        Focus only on identifying components like FBS, B27, N2, etc., mentioned in the text.

        Protocol Document:
        {document_text}

        Respond ONLY with the serum/supplement information in the following format:
            "serums_supplements": "[List of components used, e.g., 'FBS, B27']"
        If no serums are used:
            "serums_supplements": "[List of supplements used, e.g., 'N2, B27']"
        Or if no supplements are used:
            "serums_supplements": "[List of serums used, e.g., 'FBS, BSA']"
        Or if none are identified:
            "serums_supplements": "No serums or supplements used"
        """
        
        updated_steps = []
        
        for step in bm_steps:
            updated_step = step.copy()
            
            document_text = step.get('source_documents', '')
            prompt_template = ChatPromptTemplate.from_template(serums_prompt)
            prompt = prompt_template.format(document_text=document_text)
            model = Ollama(model=PROTOCOL_MODEL)
            result = model.invoke(prompt)
            
            serums = "No serums or supplements used"
            for line in result.splitlines():
                line = line.strip().strip('"').strip(',')
                if line.startswith('"serums_supplements":'):
                    serums = line.split(':', 1)[1].strip().strip('"')
                    break
                    
            updated_step["serums_supplements"] = serums
            updated_steps.append(updated_step)
        
        return updated_steps
    
    def growth_factors(ss_steps):
        growth_factors_prompt = """
        You are an expert in stem cell biology analyzing protocol documents. 
        Analyze the following protocol document to identify growth factors used in this step.
        Focus specifically on components like BMP4, FGF2, EGF, Wnt3a, etc.

        Protocol Document:
        {document_text}

        Respond ONLY with the growth factor information in the following format:
            "growth_factors": "[List of growth factors used, e.g., 'BMP4, FGF2']"
        If none are identified:
            "growth_factors": "No growth factors used"
        """
        
        updated_steps = []
        
        for step in ss_steps:
            updated_step = step.copy()
            
            document_text = step.get('source_documents', '')
            prompt_template = ChatPromptTemplate.from_template(growth_factors_prompt)
            prompt = prompt_template.format(document_text=document_text)
            model = Ollama(model=PROTOCOL_MODEL)
            result = model.invoke(prompt)
            
            factors = "No growth factors used"
            for line in result.splitlines():
                line = line.strip().strip('"').strip(',')
                if line.startswith('"growth_factors":'):
                    factors = line.split(':', 1)[1].strip().strip('"')
                    break
                    
            updated_step["growth_factors"] = factors
            updated_steps.append(updated_step)
        
        return updated_steps
    
    def cytokines_supplements(gf_steps):
        cytokines_prompt = """
        You are an expert in stem cell biology analyzing protocol documents. 
        Analyze the following protocol document to identify cytokines and chemical supplements used in this step.
        Focus specifically on components like IL-6, TGF-β, LIF, small molecules, etc.

        Protocol Document:
        {document_text}

        Respond ONLY with the cytokine/supplement information in the following format:
            "cytokines_supplements": "[List of components used, e.g., 'IL-6, CHIR99021']"
        If none are identified:
            "cytokines_supplements": "No cytokines or supplements used"
        """
        
        updated_steps = []
        
        for step in gf_steps:
            updated_step = step.copy()
            
            document_text = step.get('source_documents', '')
            prompt_template = ChatPromptTemplate.from_template(cytokines_prompt)
            prompt = prompt_template.format(document_text=document_text)
            model = Ollama(model=PROTOCOL_MODEL)
            result = model.invoke(prompt)
            
            cytokines = "No cytokines or supplements used"
            for line in result.splitlines():
                line = line.strip().strip('"').strip(',')
                if line.startswith('"cytokines_supplements":'):
                    cytokines = line.split(':', 1)[1].strip().strip('"')
                    break
                    
            updated_step["cytokines_supplements"] = cytokines
            updated_steps.append(updated_step)
        
        return updated_steps

    def passaging(cs_steps):
        passaging_prompt = """
        You are an expert in stem cell biology analyzing protocol documents. 
        Analyze the following protocol document to determine if this step involves cell passaging.
        Look for terms like: "passage", "split", "subculture", "dissociation", or specific enzymes like "Trypsin", "Accutase".

        Protocol Document:
        {document_text}

        Respond ONLY with the passaging determination in the following format:
            "passaging": "[Yes/No]"
        """
        
        updated_steps = []
        
        for step in cs_steps:
            updated_step = step.copy()
            
            document_text = step.get('source_documents', '')
            prompt_template = ChatPromptTemplate.from_template(passaging_prompt)
            prompt = prompt_template.format(document_text=document_text)
            model = Ollama(model=PROTOCOL_MODEL)
            result = model.invoke(prompt)
            
            passaging = "No"  
            for line in result.splitlines():
                line = line.strip().strip('"').strip(',')
                if line.startswith('"passaging":'):
                    passaging = line.split(':', 1)[1].strip().strip('"')
                    break
                    
            updated_step["passaging"] = passaging
            updated_steps.append(updated_step)
        
        return updated_steps

    def gene_markers(passaging_steps):
        gene_markers_prompt = """
        You are an expert in stem cell biology analyzing protocol documents. 
        Analyze the following protocol document to identify gene markers associated with this step.
        Focus on markers like:
        - Pluripotency markers (OCT4, SOX2, NANOG)
        - Lineage-specific markers (cTNT for cardiomyocytes, PAX6 for neurons)
        - Surface markers (SSEA-4, TRA-1-60)
        - Reporter genes (GFP, RFP if used for selection)

        Protocol Document:
        {document_text}

        Respond ONLY with the marker information in the following format:
            "gene_markers": "[List of gene markers, e.g., 'OCT4, cTNT']"
        If none are identified:
            "gene_markers": "No gene markers mentioned"
        """
        
        updated_steps = []
        
        for step in passaging_steps:
            updated_step = step.copy()
            
            document_text = step.get('source_documents', '')
            prompt_template = ChatPromptTemplate.from_template(gene_markers_prompt)
            prompt = prompt_template.format(document_text=document_text)
            model = Ollama(model=PROTOCOL_MODEL)
            result = model.invoke(prompt)
            
            markers = "No gene markers mentioned"
            for line in result.splitlines():
                line = line.strip().strip('"').strip(',')
                if line.startswith('"gene_markers":'):
                    markers = line.split(':', 1)[1].strip().strip('"')
                    break
                    
            updated_step["gene_markers"] = markers
            updated_steps.append(updated_step)
        
        return updated_steps
        


__all__ = ["calculate_durations",
           "basic_media",
           "serums_supplements",
           "growth_factors",
           "cytokines_supplements",
           "passaging",
           "gene_markers"]