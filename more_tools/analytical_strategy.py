import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from basic_tools.config import MODEL
from more_tools import SimilarityMethods

model="llama3"


def analytical_retrieval_strategy(query, k):
    
    print(f"Executing Analytical retrieval strategy for: '{query}'")
    
    # Define the system prompt to guide the AI in generating sub-questions
    system_prompt = """
    You are an expert at breaking down complex questions.
    Generate sub-questions that explore different aspects of the main analytical query.
    These sub-questions should cover the breadth of the topic and help retrieve 
    comprehensive information.

    Return a list of exactly 3 sub-questions, one per line.
    """

    # Create the user prompt with the main query
    user_prompt = f"Generate sub-questions for this analytical query: {query}"
    
    # Generate the sub-questions using Ollama
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options={"temperature": 0.3}
    )
    
    # Extract and clean the sub-questions
    sub_queries = response['message']['content'].strip().split('\n')
    sub_queries = [q.strip() for q in sub_queries if q.strip()]
    print(f"Generated sub-queries: {sub_queries}")
    
    # Retrieve documents for each sub-query
    all_results = []
    for sub_query in sub_queries:
        # Create embeddings for the sub-query
        sub_query_embedding = OllamaEmbeddings(sub_query, model=MODEL)
        # Perform similarity search for the sub-query
        results = SimilarityMethods.analytical_sub_similarity(sub_query_embedding, k=k/2)
        all_results.extend(results)
    
    # Ensure diversity by selecting from different sub-query results
    # Remove duplicates (same text content)
    unique_texts = set()
    diverse_results = []
    
    for result in all_results:
        if result["text"] not in unique_texts:
            unique_texts.add(result["text"])
            diverse_results.append(result)
    
    # If we need more results to reach k, add more from initial results
    if len(diverse_results) < k:
        # Direct retrieval for the main query
        main_query_embedding = OllamaEmbeddings(query, model=MODEL)
        main_results = SimilarityMethods.analytical_main_similarity(main_query_embedding, k=k)
        
        for result in main_results:
            if result["text"] not in unique_texts and len(diverse_results) < k:
                unique_texts.add(result["text"])
                diverse_results.append(result)
    
    # Return the top k diverse results
    return diverse_results[:k]