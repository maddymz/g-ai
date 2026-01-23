from sklearn.feature_extraction.text import CountVectorizer

sentences = ["I love cats", "I hate dogs"]
vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b')  # Adjusted pattern to include single characters
bow_matrix = vectorizer.fit_transform(sentences)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("Vectors:\n", bow_matrix.toarray())