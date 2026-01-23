
import numpy as np

def softmax(x):
    """
    Compute the softmax of vector x in a numerically stable way.
    """
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

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
vocab = list(vocab)  # Convert set to list to have a fixed order

# Create word-to-index and index-to-word mappings.
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for idx, word in enumerate(vocab)}

print("\nVocabulary (word to index mapping):")
for word, idx in word2idx.items():
    print(f" {word}: {idx}")

# ---------------------------------------------------
# Step 4: Generate Training Data for CBOW
# In CBOW, given the context words, we try to predict the center word.
# For each word in a sentence, the context is defined as the words
# within a window size (excluding the center word itself).
# ---------------------------------------------------
window_size = 1
training_pairs = []  # Each element is a tuple: (context_indices, center_index)

for sentence in sentences:
    for idx, word in enumerate(sentence):
        center_index = word2idx[word]
        context_indices = []
        # Collect words before the center word
        for i in range(max(0, idx - window_size), idx):
            context_indices.append(word2idx[sentence[i]])
        # Collect words after the center word
        for i in range(idx + 1, min(len(sentence), idx + window_size + 1)):
            context_indices.append(word2idx[sentence[i]])
        # Only add pairs where there is at least one context word
        if context_indices:
            training_pairs.append((context_indices, center_index))

print("\nTraining Pairs (context word indices, center word index):")
for context_idxs, center_idx in training_pairs:
    context_words = [idx2word[idx] for idx in context_idxs]
    center_word = idx2word[center_idx]
    print(f" Context: {context_words}, Center: {center_word} ({center_idx})")

# ---------------------------------------------------
# Step 5: Initialize Hyperparameters and Weights
# ---------------------------------------------------
embedding_dim = 10        # Dimension of the embedding vector
learning_rate = 0.01      # Learning rate for gradient descent
epochs = 100              # Number of epochs to train
vocab_size = len(vocab)   # Number of unique words in the vocabulary

# Weight matrices:
# W1: shape (vocab_size, embedding_dim) maps a one-hot vector to an embedding.
# W2: shape (embedding_dim, vocab_size) maps the hidden representation to scores over vocabulary.
W1 = np.random.rand(vocab_size, embedding_dim)
W2 = np.random.rand(embedding_dim, vocab_size)

# ---------------------------------------------------
# Step 6: Training the CBOW Model
# ---------------------------------------------------
print("\nStarting CBOW training...\n")
for epoch in range(epochs):
    loss_epoch = 0  # Accumulate loss over the epoch

    # Process each training pair
    for context_indices, center_idx in training_pairs:
        # ---------- Forward Pass ----------
        # 1. Look up embeddings for each context word from W1
        context_embeddings = np.array([W1[idx] for idx in context_indices])
        
        # 2. Compute the hidden layer representation by averaging the context embeddings
        h = np.mean(context_embeddings, axis=0)  # Shape: (embedding_dim,)
        
        # 3. Compute the scores over the vocabulary using W2
        scores = np.dot(h, W2)  # Shape: (vocab_size,)
        
        # 4. Apply softmax to obtain predicted probabilities
        y_pred = softmax(scores)
        
        # 5. Compute the loss (negative log likelihood for the true center word)
        loss = -np.log(y_pred[center_idx] + 1e-7)  # Adding epsilon to avoid log(0)
        loss_epoch += loss

        # ---------- Backward Pass ----------
        # 1. Compute error: the derivative of the loss with respect to the scores
        y_true = np.zeros(vocab_size)
        y_true[center_idx] = 1
        error = y_pred - y_true  # Shape: (vocab_size,)
        
        # 2. Compute gradient for W2 as the outer product of h and the error
        grad_W2 = np.outer(h, error)  # Shape: (embedding_dim, vocab_size)
        
        # 3. Compute gradient with respect to the hidden representation h
        grad_h = np.dot(W2, error)  # Shape: (embedding_dim,)
        
        # 4. Since h is the average of the context embeddings,
        #    distribute the gradient equally among them.
        grad_context = grad_h / len(context_indices)
        
        # ---------- Update Weights ----------
        # Update W1 for each context word in the training pair.
        for idx in context_indices:
            W1[idx] -= learning_rate * grad_context
        
        # Update W2
        W2 -= learning_rate * grad_W2

    # Print the average loss every 10 epochs for monitoring.
    if (epoch + 1) % 10 == 0:
        avg_loss = loss_epoch / len(training_pairs)
        print(f"Epoch {epoch + 1}/{epochs} - Average Loss: {avg_loss:.4f}")

print("\nCBOW Training complete!")

# ---------------------------------------------------
# Step 7: Display the Learned Embeddings
# ---------------------------------------------------
print("\nLearned Word Embeddings (from W1):")
for word, idx in word2idx.items():
    print(f" {word}: {W1[idx]}")