import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from basic_tools.config import MODEL
from more_tools import SimilarityMethods

model="llama3"


def opinion_retrieval_strategy(query, k):
    
    print(f"Executing Opinion retrieval strategy for: '{query}'")
    
    # Define the system prompt to guide the AI in identifying different perspectives
    system_prompt = """
        You are an expert at identifying different perspectives on a topic.
        For the given query about opinions or viewpoints, identify different perspectives 
        that people might have on this topic.

        Return a list of exactly 3 different viewpoint angles, one per line.
        """

    # Create the user prompt with the main query
    user_prompt = f"Identify different perspectives on: {query}"
    
    # Generate the different perspectives using Ollama
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options={"temperature": 0.3}
    )
    
    # Extract and clean the viewpoints
    viewpoints = response['message']['content'].strip().split('\n')
    viewpoints = [v.strip() for v in viewpoints if v.strip()]
    print(f"Identified viewpoints: {viewpoints}")
    
    # Retrieve documents representing each viewpoint
    all_results = []
    for viewpoint in viewpoints:
        # Combine the main query with the viewpoint
        combined_query = f"{query} {viewpoint}"
        # Create embeddings for the combined query
        embedding_model = OllamaEmbeddings(model=MODEL)
        viewpoint_embedding = embedding_model.embed_query(combined_query)
        # Perform similarity search for the combined query
        results = SimilarityMethods.opinion_similarity(viewpoint_embedding, k=k/2)
        
        # Mark results with the viewpoint they represent
        for result in results:
            result["viewpoint"] = viewpoint
        
        # Add the results to the list of all results
        all_results.extend(results)
    
    # Select a diverse range of opinions
    # Ensure we get at least one document from each viewpoint if possible
    selected_results = []
    for viewpoint in viewpoints:
        # Filter documents by viewpoint
        viewpoint_docs = [r for r in all_results if r.get("viewpoint") == viewpoint]
        if viewpoint_docs:
            selected_results.append(viewpoint_docs[0])
    
    # Fill remaining slots with highest similarity docs
    remaining_slots = k - len(selected_results)
    if remaining_slots > 0:
        # Sort remaining docs by similarity
        remaining_docs = [r for r in all_results if r not in selected_results]
        remaining_docs.sort(key=lambda x: x["similarity"], reverse=True)
        selected_results.extend(remaining_docs[:remaining_slots])
    
    # Return the top k results
    return selected_results[:k]