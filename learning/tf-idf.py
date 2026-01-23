import math

# Sample documents
documents = [
    "The quick brown fox jumps over the lazy dog",
    "Never jump over the lazy dog quickly",
    "A fast brown fox leaps over a lazy dog"
]

def tokenize(document):
    """
    Simple tokenizer that lowercases and splits on whitespace.
    Removes punctuation for simplicity.
    """
    # Remove punctuation
    punctuations = '.,!?;:()[]{}\'"'
    for p in punctuations:
        document = document.replace(p, '')
    # Lowercase and split
    tokens = document.lower().split()
    return tokens

def compute_tf(doc_tokens):
    """
    Computes term frequency for a single document.
    Returns a dictionary of term frequencies.
    """
    tf = {}
    for term in doc_tokens:
        tf[term] = tf.get(term, 0) + 1
    # Optionally, normalize TF by the total number of terms in the document
    total_terms = len(doc_tokens)
    for term in tf:
        tf[term] = tf[term] / total_terms
    return tf

def compute_df(documents_tokens):
    """
    Computes document frequency for all terms in the corpus.
    Returns a dictionary of document frequencies.
    """
    df = {}
    for tokens in documents_tokens:
        unique_terms = set(tokens)
        for term in unique_terms:
            df[term] = df.get(term, 0) + 1
    return df

def compute_idf(df, total_docs):
    """
    Computes inverse document frequency for all terms.
    Returns a dictionary of IDF scores.
    """
    idf = {}
    for term, freq in df.items():
        idf[term] = math.log(total_docs / (1 + freq)) + 1  # Adding 1 to avoid division by zero
    return idf

def compute_tf_idf(tf, idf):
    """
    Computes TF-IDF for a single document.
    Returns a dictionary of TF-IDF scores.
    """
    tf_idf = {}
    for term, tf_value in tf.items():
        tf_idf[term] = tf_value * idf.get(term, 0)
    return tf_idf

def main(documents):
    # Step 1: Tokenize all documents
    documents_tokens = [tokenize(doc) for doc in documents]
    
    # Step 2: Compute TF for each document
    tfs = [compute_tf(tokens) for tokens in documents_tokens]
    
    # Step 3: Compute DF across all documents
    df = compute_df(documents_tokens)
    
    # Step 4: Compute IDF for all terms
    total_docs = len(documents)
    idf = compute_idf(df, total_docs)
    
    # Step 5: Compute TF-IDF for each document
    tf_idfs = [compute_tf_idf(tf, idf) for tf in tfs]
    
    # (Optional) Collect all unique terms for creating a TF-IDF matrix
    all_terms = sorted(df.keys())
    
    # Display TF-IDF scores
    for i, tf_idf in enumerate(tf_idfs):
        print(f"\nDocument {i+1} TF-IDF:")
        for term in all_terms:
            score = tf_idf.get(term, 0)
            if score > 0:
                print(f"  {term}: {score:.4f}")

if __name__ == "__main__":
    main(documents)