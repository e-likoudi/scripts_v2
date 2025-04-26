import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from sentence_transformers import CrossEncoder
from basic_tools.config import MODEL
from more_tools import SimilarityMethods

cross_encoder = CrossEncoder("cross-encoder/stsb-roberta-large")
model="llama3"

def contextual_retrieval_strategy(query, k, user_context=None):
    
    print(f"Executing Contextual retrieval strategy for: '{query}'")
    
    # If no user context provided, try to infer it from the query
    if not user_context:
        system_prompt = """
        You are an expert at understanding implied context in questions.
        For the given query, infer what contextual information might be relevant or implied 
        but not explicitly stated. Focus on what background would help answering this query.

        Return a brief description of the implied context.
        """

        user_prompt = f"Infer the implied context in this query: {query}"
        
        # Generate the inferred context using Ollama
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0.1}
        )
        
        # Extract and print the inferred context
        user_context = response['message']['content'].strip()
        print(f"Inferred context: {user_context}")
    
    # Reformulate the query to incorporate context
    system_prompt = """
    You are an expert at reformulating questions with context.
    Given a query and some contextual information, create a more specific query that 
    incorporates the context to get more relevant information.

    Return ONLY the reformulated query without explanation.
    """

    user_prompt = f"""
    Query: {query}
    Context: {user_context}

    Reformulate the query to incorporate this context:
    """
    
    # Generate the contextualized query using Ollama
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options={"temperature": 0}
    )
    
    # Extract and print the contextualized query
    contextualized_query = response['message']['content'].strip()
    print(f"Contextualized query: {contextualized_query}")
    
    # Retrieve documents based on the contextualized query
    query_embedding = OllamaEmbeddings(contextualized_query,model=MODEL)
    initial_results = SimilarityMethods.contexual_similarity(query_embedding, k=k*2)
    
    # Rank documents considering both relevance and user context
    ranked_results = []
    
    for doc in initial_results:
        # Score document relevance considering the context
        context_relevance = cross_encoder(query, user_context, doc["text"])
        ranked_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "similarity": doc["similarity"],
            "context_relevance": context_relevance
        })
    
    # Sort by context relevance and return top k results
    ranked_results.sort(key=lambda x: x["context_relevance"], reverse=True)
    return ranked_results[:k]