import numpy as np

def softmax(x):
    """
    Compute the softmax of vector x.
    We subtract the maximum value for numerical stability.
    """
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

# -------------------------
# Step 1: Define the Corpus
# -------------------------
corpus = [
    "I like deep learning",
    "I like NLP",
    "I enjoy flying"
]

print("Original Corpus:")
for sentence in corpus:
    print(" -", sentence)

# ------------------------------------
# Step 2: Preprocess the Corpus
# Lowercase and tokenize each sentence.
# ------------------------------------
sentences = [sentence.lower().split() for sentence in corpus]
print("\nTokenized Sentences:")
for sentence in sentences:
    print(" -", sentence)

# -----------------------------------------
# Step 3: Build the Vocabulary and Mappings
# -----------------------------------------
vocab = set()  # use a set to avoid duplicates
for sentence in sentences:
    for word in sentence:
        vocab.add(word)
vocab = list(vocab)  # convert to list to fix ordering

# Create dictionaries to map words to indices and vice-versa.
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for idx, word in enumerate(vocab)}

print("\nVocabulary (word to index mapping):")
for word, idx in word2idx.items():
    print(f" {word}: {idx}")

# -------------------------------------------------------
# Step 4: Generate Training Data (Skip-gram Pairs)
# -------------------------------------------------------
# For each word in a sentence, use a window of size 1 to collect context words.
window_size = 1
training_pairs = []  # will store tuples of (center_word_idx, context_word_idx)

for sentence in sentences:
    for idx, word in enumerate(sentence):
        center_word_idx = word2idx[word]
        # Determine the indices for the context window
        context_indices = list(range(max(0, idx - window_size), idx)) + \
                          list(range(idx + 1, min(len(sentence), idx + window_size + 1)))
        for context_idx in context_indices:
            context_word_idx = word2idx[sentence[context_idx]]
            training_pairs.append((center_word_idx, context_word_idx))

print("\nTraining Pairs (center word index, context word index):")
for center, context in training_pairs:
    print(f" Center: {idx2word[center]} ({center}), Context: {idx2word[context]} ({context})")

# ------------------------------------------------------
# Step 5: Initialize Hyperparameters and Weight Matrices
# ------------------------------------------------------
embedding_dim = 10       # size of the embedding vector
learning_rate = 0.01     # learning rate for SGD updates
epochs = 100             # number of epochs for training
vocab_size = len(vocab)  # number of unique words

# Weight matrices:
# W1: shape (vocab_size, embedding_dim) - maps one-hot input to embeddings
# W2: shape (embedding_dim, vocab_size) - maps embeddings to scores over vocabulary
W1 = np.random.rand(vocab_size, embedding_dim)
W2 = np.random.rand(embedding_dim, vocab_size)

# --------------------------------
# Step 6: Training the Model
# --------------------------------
print("\nStarting training...\n")
for epoch in range(epochs):
    loss_epoch = 0  # accumulate loss over the epoch
    
    # Iterate through each training pair
    for center_idx, context_idx in training_pairs:
        # ---------- Forward Pass ----------
        # 1. Look up the embedding for the center word (from W1)
        center_embedding = W1[center_idx]  # shape: (embedding_dim,)
        
        # 2. Compute scores for all words by multiplying the embedding with W2
        scores = np.dot(center_embedding, W2)  # shape: (vocab_size,)
        
        # 3. Apply softmax to get probabilities over the vocabulary
        y_pred = softmax(scores)  # shape: (vocab_size,)
        
        # 4. Compute the loss (negative log likelihood for the true context word)
        loss = -np.log(y_pred[context_idx] + 1e-7)  # add a small number to prevent log(0)
        loss_epoch += loss

        # ---------- Backward Pass ----------
        # Create a one-hot encoded vector for the true context word
        y_true = np.zeros(vocab_size)
        y_true[context_idx] = 1
        
        # Compute the error: derivative of loss with respect to the scores
        error = y_pred - y_true  # shape: (vocab_size,)
        
        # Compute gradients for W2 and the center embedding:
        # Gradient for W2 is the outer product of the center embedding and the error
        grad_W2 = np.outer(center_embedding, error)  # shape: (embedding_dim, vocab_size)
        
        # Gradient for the center embedding (W1 row) is the dot product of W2 and the error
        grad_center = np.dot(W2, error)  # shape: (embedding_dim,)
        
        # ---------- Update Weights ----------
        # Update the embedding for the center word in W1
        W1[center_idx] -= learning_rate * grad_center
        
        # Update W2 with the computed gradient
        W2 -= learning_rate * grad_W2

    # Print the average loss every 10 epochs for monitoring
    if (epoch + 1) % 10 == 0:
        avg_loss = loss_epoch / len(training_pairs)
        print(f"Epoch {epoch + 1}/{epochs} - Average Loss: {avg_loss:.4f}")

print("\nTraining complete!")

# --------------------------------------
# Step 7: Display the Learned Embeddings
# --------------------------------------
print("\nLearned Word Embeddings (from W1):")
for word, idx in word2idx.items():
    print(f" {word}: {W1[idx]}")