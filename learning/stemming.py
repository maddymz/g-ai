def simple_stem(word):
    suffixes = ["ing", "ly", "ed", "ious", "ies", "ive", "es", "s", "ment"]
    for suffix in suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)]  # Remove the matched suffix.
    return word

# Example usage
words = ["running", "happily", "tried", "faster", "cats"]
stemmed_words = [simple_stem(word) for word in words]
print("Stemmed Words:", stemmed_words)