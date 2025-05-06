import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ollama

model="llama3"

def classify_query(query):

    # Define the system prompt to guide the AI's classification
    system_prompt = """
        You are an expert at classifying questions. 
        Classify the given query into exactly one of these categories:
        - Factual: Queries seeking specific, verifiable information.
        - Analytical: Queries requiring comprehensive analysis or explanation.
        - Opinion: Queries about subjective matters or seeking diverse viewpoints.
        - Contextual: Queries that depend on user-specific context.

        Return ONLY the category name, without any explanation or additional text.
    """

    # Create the user prompt with the query to be classified
    user_prompt = f"Classify this query: {query}"
    
    # Generate the classification response from the Ollama model
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options={
            "temperature": 0
        }
    )
    
    # Extract and strip the category from the response
    category = response['message']['content'].strip()
    
    # Define the list of valid categories
    valid_categories = ["Factual", "Analytical", "Opinion", "Contextual"]
    
    # Ensure the returned category is valid
    for valid in valid_categories:
        if valid in category:
            return valid
    
    # Default to "Factual" if classification fails
    return "Factual"

# Test cases
"""
test_queries = [
    "What is the capital of France?",                          # Should be Factual
    "Explain the causes of World War II",                      # Should be Analytical
    "Do you think chocolate ice cream is better than vanilla?", # Should be Opinion
    "Based on my purchase history, what should I buy next?",    # Should be Contextual
    "How many planets are in our solar system?"                # Should be Factual
]

# Run tests
for query in test_queries:
    category = classify_query(query)
    print(f"Query: '{query}'\nCategory: {category}\n")
"""