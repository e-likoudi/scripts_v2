import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ollama
from langchain_core.documents import Document  
from langchain_community.embeddings.ollama import OllamaEmbeddings
from sentence_transformers import CrossEncoder
from basic_tools.config import MODEL
from more_tools import SimilarityMethods

model="llama3"

def factual_retrieval_strategy(query, k):
    
    print(f"Executing Factual retrieval strategy for: '{query}'")
    
    # Use LLM to enhance the query for better precision
    system_prompt = """
        You are an expert at enhancing search queries.
        Your task is to reformulate the given factual query to make it more precise and 
        specific for information retrieval. Focus on key entities and their relationships.

        Provide ONLY the enhanced query without any explanation.
    """

    user_prompt = f"Enhance this factual query: {query}"
    
    # Generate the enhanced query using Ollama
    response = ollama.chat(
        model=model,  # Using default Ollama model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options={"temperature": 0}
    )
    
    # Extract and print the enhanced query
    enhanced_query = response['message']['content'].strip()
    print(f"Enhanced query: {enhanced_query}")
    
    # Create embeddings for the enhanced query
    embedding_model = OllamaEmbeddings(model=MODEL)
    query_embedding = embedding_model.embed_query(enhanced_query)
    
    # Perform initial similarity search to retrieve documents
    initial_results = SimilarityMethods.factual_similarity(query_embedding, k)
    
    # Initialize a list to store ranked results
    ranked_results = []
    
    # Score and rank documents by relevance using LLM
    for doc in initial_results:
        # Proper way to handle both Document objects and dictionaries
        if isinstance(doc, Document):
            doc_text = doc.page_content
            doc_metadata = doc.metadata
            doc_similarity = getattr(doc, 'similarity', 0.0)  # Safe attribute access
        else:
            # Handle case where doc might be a dictionary
            doc_text = doc.get('text', '') if hasattr(doc, 'get') else str(doc)
            doc_metadata = doc.get('metadata', {}) if hasattr(doc, 'get') else {}
            doc_similarity = doc.get('similarity', 0.0) if hasattr(doc, 'get') else 0.0
        
        relevance_score = score_document_relevance(enhanced_query, doc_text)
        ranked_results.append({
            "text": doc_text,
            "metadata": doc_metadata,
            "similarity": doc_similarity,
            "relevance_score": relevance_score
        })

    # Sort the results by relevance score in descending order
    ranked_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    # Return the top k results
    return ranked_results[:k]

def score_document_relevance(query: str, document_text: str) -> float:
    """
    Scores document relevance using a pre-trained CrossEncoder model.
    More efficient and accurate than LLM-based scoring for this task.
    
    Args:
        query: Search query
        document_text: Document text to evaluate
        
    Returns:
        Relevance score between 0 (irrelevant) and 1 (highly relevant)
    """
    # Initialize model (loaded once and cached)
    model = CrossEncoder("cross-encoder/stsb-roberta-large")
    
    # Score the pair
    score = model.predict([(query, document_text)])[0]
    return float(score)