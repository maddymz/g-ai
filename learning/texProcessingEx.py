# Sample text containing various cases
text = "Apple released the iPhone! I didn't know that Apple's announcement would shock everyone. Don't you think it's amazing?"

print("Original Text:")
print(text)
print("-" * 100)

# 1. Lowercasing: Convert all text to lowercase
lower_text = text.lower()
print("After Lowercasing:")
print(lower_text)
print("-" * 100)

# 2. Tokenization: Split text into words (this simple approach splits on whitespace)
tokens = lower_text.split()
print("After Tokenization:")
print(tokens)
print("-" * 100)

# 3. Stripping Punctuation: Remove punctuation from each token
# Define a set of punctuation characters
punctuations = '.,!?\'":;()'
tokens = [token.strip(punctuations) for token in tokens]
print("After Removing Punctuation:")
print(tokens)
print("-" * 100)

# 4. Removing Stop Words: Filter out common, semantically insignificant words
stop_words = ['the', 'is', 'at', 'on', 'and', 'a', 'an', 'of', 'that', 'would', 'you', 'it']
tokens = [token for token in tokens if token not in stop_words]
print("After Removing Stop Words:")
print(tokens)
print("-" * 100)

# 5. Expanding Contractions: Replace contractions with their expanded forms
# Note: This is a simple dictionary for demonstration
contractions = {
    "didn't": "did not",
    "don't": "do not",
    "it's": "it is",
    "i'm": "i am",
    "i've": "i have",
    "apple's": "apple has"
}

expanded_tokens = []
for token in tokens:
    if token in contractions:
        # Split the expanded form to keep tokens consistent
        expanded_tokens.extend(contractions[token].split())
    else:
        expanded_tokens.append(token)
tokens = expanded_tokens
print("After Expanding Contractions:")
print(tokens)
print("-" * 100)

# 6. Handling Special Characters and Numbers:
# For this example, remove tokens that are purely numeric.
tokens = [token for token in tokens if not token.isdigit()]
print("After Handling Numbers:")
print(tokens)
print("-" * 100)

# 7. Correcting Misspellings:
# A very basic approach using a predefined dictionary of common corrections.
corrections = {
    "iphon": "iphone",  # Example: if a typo occurred
    # add more common misspellings as needed
}
tokens = [corrections.get(token, token) for token in tokens]
print("After Correcting Misspellings:")
print(tokens)
print("-" * 100)

# 8. Dealing with Abbreviations and Acronyms:
# Expand or standardize abbreviations using a simple mapping.
abbreviations = {
    "ai": "artificial intelligence",
    # add additional abbreviation mappings as needed
}
tokens = [abbreviations.get(token, token) for token in tokens]
print("After Expanding Abbreviations:")
print(tokens)
print("-" * 100)

# Final preprocessed tokens
print("Final Preprocessed Tokens:")
print(tokens)