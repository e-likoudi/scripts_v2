import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from more_tools.factual_strategy import factual_retrieval_strategy
from more_tools.analytical_strategy import analytical_retrieval_strategy
from more_tools.opinion_strategy import opinion_retrieval_strategy
from more_tools.contexual_strategy import contextual_retrieval_strategy
from more_tools.classify_query import classify_query

model="llama3"

def adaptive_retrieval(query, k=4, user_context=None):
   
    # Classify the query to determine its type
    query_type = classify_query(query)
    print(f"Query classified as: {query_type}")
    
    # Select and execute the appropriate retrieval strategy based on the query type
    if query_type == "Factual":
        # Use the factual retrieval strategy for precise information
        results = factual_retrieval_strategy(query, k)
    elif query_type == "Analytical":
        # Use the analytical retrieval strategy for comprehensive coverage
        results = analytical_retrieval_strategy(query, k)
    elif query_type == "Opinion":
        # Use the opinion retrieval strategy for diverse perspectives
        results = opinion_retrieval_strategy(query, k)
    elif query_type == "Contextual":
        # Use the contextual retrieval strategy, incorporating user context
        results = contextual_retrieval_strategy(query, k, user_context)
    else:
        # Default to factual retrieval strategy if classification fails
        results = factual_retrieval_strategy(query, k)
    
    return results  # Return the retrieved documents