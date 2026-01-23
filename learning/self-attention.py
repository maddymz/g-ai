
import numpy as np

# Sample input: a sentence represented as word embeddings
sentence = np.array([
    [0.1, 0.2, 0.3, 0.4],   # "it"
    [0.5, 0.6, 0.7, 0.8],   # "refers"
    [0.9, 1.0, 1.1, 1.2],   # "to"
    [1.3, 1.4, 1.5, 1.6],   # "robot"
    [1.7, 1.8, 1.9, 2.0]    # "."
])

def self_attention(query, keys, values):
    """
    Demonstrates self-attention for one query with detailed outputs at each step.
    
    Parameters:
        query: A single query vector.
        keys: Multiple key vectors.
        values: Multiple value vectors.
    
    Returns:
        output: The weighted sum of the values (the attention output).
        attention_weights: The computed attention weights.
    """
    print("=== Self-Attention Computation ===")
    
    # Step 1: Dot Product between Query and Keys
    print("\nStep 1: Dot Product (Similarity Calculation)")
    print("Original query:", query)
    query = query[np.newaxis, :]  # Reshape query for matrix multiplication
    print("Reshaped query (for multiplication):", query)
    
    keys_transposed = keys.T      # Transpose keys for proper alignment
    print("Transposed keys:\n", keys_transposed)
    
    dot_product = np.dot(query, keys_transposed)
    print("Resulting dot product:", dot_product)
    
    # Step 2: Apply Softmax to Convert Dot Products to Probabilities
    print("\nStep 2: Softmax Normalization")
    exp_dot_product = np.exp(dot_product)
    print("Exponentiated dot product:", exp_dot_product)
    
    sum_exp = exp_dot_product.sum(axis=1, keepdims=True)
    print("Sum of exponentiated scores:", sum_exp)
    
    attention_weights = exp_dot_product / sum_exp
    print("Attention weights after softmax normalization:", attention_weights)
    
    # Step 3: Weighted Sum of Values to Get the Output
    print("\nStep 3: Weighted Sum of Values (Creating the Output)")
    output = np.dot(attention_weights, values)
    print("Output vector (weighted sum of values):", output)
    
    return output, attention_weights

# "it" is our query; the words "refers", "to", "robot" act as keys and values.
query = sentence[0]          # "it"
keys = sentence[1:-1]        # "refers", "to", "robot"
values = sentence[1:-1]      # In this simple example, keys and values are the same

# Perform self-attention computation
output, attn_weights = self_attention(query, keys, values)

print("\n=== Final Results ===")
print("Final Output of Self-Attention:", output)
print("Final Attention Weights:", attn_weights)