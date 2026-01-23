# Toy dataset
sentences = [
    "I love natural language processing",
    "Language models are amazing"
]

# Function to generate bigrams
def generate_bigrams(sentence):
    words = sentence.lower().split()  # Tokenization (lowercase + split)
    bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    return bigrams, words  # Return bigrams and word list

# Collect all words and bigrams
all_bigrams = []
all_words = set()  # To store unique words

for sentence in sentences:
    bigrams, words = generate_bigrams(sentence)
    all_bigrams.extend(bigrams)
    all_words.update(words)

# Sort words for consistent ordering
unique_words = sorted(all_words)

# Create bigram frequency matrix
bigram_matrix = {word: {w: 0 for w in unique_words} for word in unique_words}

# Count occurrences of bigrams
for bigram in all_bigrams:
    first, second = bigram
    bigram_matrix[first][second] += 1

# Convert frequency matrix to probability matrix
for word in unique_words:
    total_bigrams = sum(bigram_matrix[word].values())  # Total transitions from this word
    if total_bigrams > 0:
        for next_word in unique_words:
            bigram_matrix[word][next_word] /= total_bigrams  # Normalize to get probabilities

# Display bigram probability matrix in a well-formatted table
print("\nBigram Probability Matrix:\n")

# Print header row
header = ["Word"] + unique_words
col_width = 8  # Set a fixed column width for better alignment

# Print table header
print(f"{header[0]:<{col_width}}", end=" | ")
print(" | ".join(f"{w:<{col_width}}" for w in header[1:]))
print("-" * (col_width * (len(unique_words) + 1) + 3))

# Print each row with bigram probabilities
for word in unique_words:
    row_values = [f"{bigram_matrix[word][w]:.1f}" for w in unique_words]
    print(f"{word:<{col_width}}", end=" | ")
    print(" | ".join(f"{val:<{col_width}}" for val in row_values))