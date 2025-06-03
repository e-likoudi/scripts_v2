import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ollama
from basic_tools.config import PROTOCOL_MODEL

def generate_summary(documents):
            
    summaries = []
    
    for i in range(0, len(documents), 2):
        batch = documents[i:i+2]
        combined_text = "\n".join(doc.page_content for doc in batch)
        
        # Generate concise summary
        summary_prompt = f"""
        You are an expert research assistant. Summarize the following academic paper section with precision, focusing on objectives, methods, and findings. 
        Maintain a formal tone.
        
        === REQUIREMENTS ===
        1. Extract key steps in chronological order
        2. Identify all critical components
        3. Note any special conditions
        4. Keep under 200 words
        
        === DOCUMENTS ===
        {combined_text}
        """
        
        response = ollama.chat(
            model=PROTOCOL_MODEL,
            messages=[{"role": "user", "content": summary_prompt}],
            options={"temperature": 0.1}
        )
        
        summaries.append(response['message']['content'].strip())
    
    return summaries
