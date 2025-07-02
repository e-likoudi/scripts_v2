import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.llms.ollama import Ollama
from basic_tools.config import PROTOCOL_MODEL

def create_protocol(cell_line, durations, enriched_steps):
    prompt = """
    You are an expert in biological protocols.
    Your task is to refine a differentiation protocol for stem cells into a clear and concise format.
    The protocol should be structured and easy to follow, with each step clearly defined and numbered.
    - Step 0: The condition of the undifferentiated cells before any differentiation starts.
    - Steps 1 through n: The differentiation process, including any specific signals, medium changes, or lineage-commitment steps.
    - The last step: The final differentiated state of the cells, including any markers or characteristics that define this state.
    
    The protocol should have the following format:
    - Cell line and target information: {cell_line}
    - Differentiation step: {stages}
    - Duration for each step: {durations}
    - Description of each step: {reason}
    - Basic media: {media}
    - Any serums or supplements used: {supplements}
    - Growth factors: {growth_factors}
    - Cytokines and supplements: {cytokines_supplements}
    - Passaging: {passaging}
    - Gene markers: {gene_markers}
    
    """

    steps_data = []
    for step in enriched_steps:
        steps_data.append({
            'stage': step.get('stage', ''),
            'duration': step.get('duration', ''),
            'description': step.get('reason', ''),
            'media': step.get('media', []),
            'supplements': step.get('serums_supplements', []),
            'growth_factors': step.get('growth_factors', []),
            'cytokines': step.get('cytokines_supplements', []),
            'passaging': step.get('passaging', ''),
            'gene_markers': step.get('gene_markers', [])
        })

    formatted_prompt = prompt.format(
        cell_line=cell_line,
        stages=steps_data['stage'],
        durations=durations,
        reason=step['description'],
        media=step['media'],
        supplements=step['supplements'],
        growth_factors=step['growth_factors'],
        cytokines_supplements=step['cytokines'],
        passaging=step['passaging'],
        gene_markers=step['gene_markers']
    )

    model = Ollama(model=PROTOCOL_MODEL)
    response = model.invoke(formatted_prompt)

    return response