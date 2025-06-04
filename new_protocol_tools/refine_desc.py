import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ollama
from basic_tools.config import PROTOCOL_MODEL

def refine_with_prompt(steps_data):
    refined_steps = []
    
    for step in steps_data:
        prompt = f"""
        Combine these protocol descriptions into one cohesive paragraph:
        
        Step {step['step_num']}: {step['name']}
        
        Input Descriptions:
        {chr(10).join(step['description'])}
        
        Requirements:
        1. Preserve all technical details (concentrations, durations)
        2. Remove redundant information
        3. Maintain chronological order
        4. Keep under 100 words
        5. Use passive voice for protocols
        """
        
        response = ollama.chat(
            model=PROTOCOL_MODEL,  
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0}
        )
        
        refined_steps.append({
            **step,
            'refined_description': response['message']['content'].strip()
        })
    
    refined_desc = '\n\n'.join(
        f"Step {s['step_num']}: {s['name']}\n"
        f"Duration: Not specified\n"
        f"{s['refined_description']}" 
        for s in refined_steps
    )

    return refined_desc.strip() if refined_desc else "No valid steps found."

