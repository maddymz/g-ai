def simple_tokenize(text):
    tokens = []
    current_word = ""
    for char in text:
        if char.isalnum():
            current_word += char
        else:
            if current_word != "":
                tokens.append(current_word)  # Append the accumulated word.
                current_word = ""
            if char.strip() != "":  # Ignore whitespace.
                tokens.append(char)  # Append punctuation or other non-alphanumeric characters.
    if current_word != "":
        tokens.append(current_word)  # Append any remaining word.
    return tokens

# Example usage
sentence = "Generative AI is fascinating!"
tokens = simple_tokenize(sentence)
print(tokens)