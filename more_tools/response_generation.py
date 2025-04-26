import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ollama


def generate_response(query, results, query_type, model="llama3"):
    """
    Generate a response based on query, retrieved documents, and query type.
    
    Args:
        query (str): User query
        results (List[Dict]): Retrieved documents
        query_type (str): Type of query
        model (str): LLM model
        
    Returns:
        str: Generated response
    """
    # Prepare context from retrieved documents by joining their texts with separators
    context = "\n\n---\n\n".join([r["text"] for r in results])
    
    # Create custom system prompt based on query type
    if query_type == "Factual":
        system_prompt = """
        You are a helpful assistant providing factual information.
        Answer the question based on the provided context. Focus on accuracy and precision.
        If the context doesn't contain the information needed, acknowledge the limitations.
        """
        
    elif query_type == "Analytical":
        system_prompt = """
        You are a helpful assistant providing analytical insights.
        Based on the provided context, offer a comprehensive analysis of the topic.
        Cover different aspects and perspectives in your explanation.
        If the context has gaps, acknowledge them while providing the best analysis possible.
        """
        
    elif query_type == "Opinion":
        system_prompt = """
        You are a helpful assistant discussing topics with multiple viewpoints.
        Based on the provided context, present different perspectives on the topic.
        Ensure fair representation of diverse opinions without showing bias.
        Acknowledge where the context presents limited viewpoints.
        """
        
    elif query_type == "Contextual":
        system_prompt = """
        You are a helpful assistant providing contextually relevant information.
        Answer the question considering both the query and its context.
        Make connections between the query context and the information in the provided documents.
        If the context doesn't fully address the specific situation, acknowledge the limitations.
        """
        
    else:
        system_prompt = """
        You are a helpful assistant. 
        Answer the question based on the provided context. 
        If you cannot answer from the context, acknowledge the limitations.
        """
    
    # Create user prompt by combining the context and the query
    user_prompt = f"""
    Context:
    {context}

    Question: {query}

    Please provide a helpful response based on the context.
    """
    
     # Generate the sub-questions using Ollama
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options={"temperature": 0.2}
    )
    
    
    # Return the generated response content
    return response.choices[0].message.content