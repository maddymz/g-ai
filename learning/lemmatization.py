def simple_lemmatize(word):
    # A minimal dictionary for known irregular forms.
    irregular_lemmas = {
        "running": "run",
        "happily": "happy",
        "ran": "run",
        "better": "good",
        "faster": "fast",
        "cats": "cat",
        "dogs": "dog",
        "are": "be",
        "is": "be",
        "have": "have"
    }
    return irregular_lemmas.get(word, word)

# Example usage
words = ["running", "happily", "ran", "better", "faster", "cats"]
lemmatized_words = [simple_lemmatize(word) for word in words]
print("Lemmatized Words:", lemmatized_words)