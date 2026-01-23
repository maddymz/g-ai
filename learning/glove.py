import numpy as np

def weighting_func(x, x_max=100, alpha=0.75):
    """
    Compute the weighting function for a co-occurrence count.
    If x < x_max, return (x / x_max)^alpha; otherwise return 1.
    """
    return (x / x_max) ** alpha if x < x_max else 1

# ---------------------------------------------------
# Step 1: Define the Corpus
# ---------------------------------------------------
corpus = [
    "I like deep learning",
    "I like NLP",
    "I enjoy flying"
]

print("Original Corpus:")
for sentence in corpus:
    print(" -", sentence)

# ---------------------------------------------------
# Step 2: Preprocess the Corpus
# Lowercase and tokenize each sentence.
# ---------------------------------------------------
sentences = [sentence.lower().split() for sentence in corpus]
print("\nTokenized Sentences:")
for sentence in sentences:
    print(" -", sentence)

# ---------------------------------------------------
# Step 3: Build the Vocabulary and Mappings
# ---------------------------------------------------
vocab = set()
for sentence in sentences:
    for word in sentence:
        vocab.add(word)
vocab = list(vocab)
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for idx, word in enumerate(vocab)}

print("\nVocabulary (word to index mapping):")
for word, idx in word2idx.items():
    print(f" {word}: {idx}")

# ---------------------------------------------------
# Step 4: Build the Co-occurrence Matrix
# We'll use a window size of 1 for simplicity.
# ---------------------------------------------------
vocab_size = len(vocab)
X = np.zeros((vocab_size, vocab_size))
window_size = 1

for sentence in sentences:
    sentence_length = len(sentence)
    for i, word in enumerate(sentence):
        word_idx = word2idx[word]
        # Define the window boundaries
        start = max(0, i - window_size)
        end = min(sentence_length, i + window_size + 1)
        for j in range(start, end):
            if i == j:
                continue  # Skip the word itself
            context_word = sentence[j]
            context_idx = word2idx[context_word]
            X[word_idx, context_idx] += 1

print("\nCo-occurrence Matrix (X):")
print(X)

# ---------------------------------------------------
# Step 5: Initialize GloVe Parameters
# ---------------------------------------------------
embedding_dim = 10      # Dimension of the embeddings
learning_rate = 0.05
epochs = 100

# Initialize word and context embeddings randomly
W = np.random.rand(vocab_size, embedding_dim)
W_context = np.random.rand(vocab_size, embedding_dim)

# Initialize bias terms for words and context words
b = np.random.rand(vocab_size)
b_context = np.random.rand(vocab_size)

# ---------------------------------------------------
# Step 6: Train the GloVe Model
# ---------------------------------------------------
# We minimize the cost: f(X_ij) * (w_i^T w_j~ + b_i + b_j~ - log(X_ij))^2
for epoch in range(epochs):
    total_cost = 0
    # Iterate over all nonzero co-occurrence entries
    for i in range(vocab_size):
        for j in range(vocab_size):
            if X[i, j] > 0:
                # Compute weighting for this co-occurrence
                weight = weighting_func(X[i, j])
                # Calculate the difference between prediction and log count
                diff = np.dot(W[i], W_context[j]) + b[i] + b_context[j] - np.log(X[i, j])
                cost = weight * (diff ** 2)
                total_cost += cost
                # Compute gradient (the factor 2 comes from the derivative of the square)
                grad = 2 * weight * diff
                # Update the parameters using gradient descent
                W[i] -= learning_rate * grad * W_context[j]
                W_context[j] -= learning_rate * grad * W[i]
                b[i] -= learning_rate * grad
                b_context[j] -= learning_rate * grad
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Total Cost: {total_cost:.4f}")

# Combine word and context embeddings as the final representation
final_embeddings = W + W_context

print("\nLearned GloVe Embeddings:")
for word, idx in word2idx.items():
    print(f" {word}: {final_embeddings[idx]}")