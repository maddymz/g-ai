#!/usr/bin/env python3
"""
Job Search Application using BERT Embeddings
Generates embeddings from job descriptions and performs semantic search.
Dataset: https://www.kaggle.com/datasets/kshitizregmi/jobs-and-job-description
"""

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import sys
import os


# Download required NLTK data
def download_nltk_resources():
    """Download required NLTK resources if not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)


def preprocess_text(text):
    """
    Preprocess text by tokenizing, removing stopwords, punctuation, and lemmatizing.
    
    Args:
        text (str): Raw text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)


def generate_embedding(text, tokenizer, model, device):
    """
    Generate BERT embedding for the given text using [CLS] token.
    
    Args:
        text (str): Text to generate embedding for
        tokenizer: BERT tokenizer
        model: BERT model
        device: torch device (cpu or cuda)
        
    Returns:
        numpy.ndarray: Embedding vector
    """
    # Tokenize with truncation to handle long descriptions
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=512,
        padding=True
    )
    
    # Move inputs to device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract [CLS] token embedding (first token)
    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    return cls_embedding[0]


def save_embeddings(path, embeddings, df):
    """Save embeddings and dataframe to a .npz file."""
    np.savez_compressed(path, embeddings=embeddings, titles=df.get('Job Title'), df_index=df.index.values)


def load_embeddings(path):
    """Load embeddings from a .npz file."""
    data = np.load(path, allow_pickle=True)
    embeddings = data['embeddings']
    titles = data.get('titles')
    index = data.get('df_index')
    return embeddings, titles, index


def semantic_search(query, job_embeddings, df, tokenizer, model, device, top_k=5):
    """
    Perform semantic search using BERT embeddings and cosine similarity.
    
    Args:
        query (str): Search query
        job_embeddings (numpy.ndarray): Precomputed job embeddings
        df (pandas.DataFrame): DataFrame with job data
        tokenizer: BERT tokenizer
        model: BERT model
        device: torch device
        top_k (int): Number of top results to return
        
    Returns:
        pandas.DataFrame: Top matching jobs
    """
    # Preprocess and generate embedding for query
    processed_query = preprocess_text(query)
    query_embedding = generate_embedding(processed_query, tokenizer, model, device)
    
    # Reshape for cosine similarity computation
    query_embedding = query_embedding.reshape(1, -1)
    
    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, job_embeddings)[0]
    
    # Get top k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Create results DataFrame
    results = df.iloc[top_indices].copy()
    results['similarity_score'] = similarities[top_indices]
    
    return results


def load_and_preprocess_dataset(csv_path):
    """
    Load and preprocess the job dataset.
    
    Args:
        csv_path (str): Path to CSV file
        
    Returns:
        pandas.DataFrame: Preprocessed DataFrame
    """
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Display dataset info
    print(f"Dataset loaded: {len(df)} jobs")
    print(f"Columns: {', '.join(df.columns.tolist())}")
    
    # Combine title and description for better context
    if 'Job Title' in df.columns and 'Job Description' in df.columns:
        df['combined_text'] = df['Job Title'].fillna('') + ' ' + df['Job Description'].fillna('')
    elif 'title' in df.columns and 'description' in df.columns:
        df['combined_text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    else:
        # Try to find appropriate columns
        text_column = df.columns[0] if len(df.columns) > 0 else None
        if text_column:
            df['combined_text'] = df[text_column].fillna('')
        else:
            raise ValueError("Could not find appropriate text columns in dataset")
    
    print("Preprocessing text data...")
    df['processed_text'] = df['combined_text'].apply(preprocess_text)
    
    return df


def main():
    """Main application logic."""
    # Check for dataset path
    if len(sys.argv) < 2:
        print("Usage: python job_search_bert.py <path_to_dataset.csv>")
        print("\nDataset: https://www.kaggle.com/datasets/kshitizregmi/jobs-and-job-description")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not os.path.exists(csv_path):
        print(f"Error: Dataset file not found: {csv_path}")
        sys.exit(1)
    
    # Download NLTK resources
    print("Downloading NLTK resources...")
    download_nltk_resources()
    
    # Load and preprocess dataset
    df = load_and_preprocess_dataset(csv_path)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load pretrained BERT model and tokenizer
    print("Loading BERT model (bert-base-uncased)...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.to(device)
    model.eval()
    
    # Embedding persistence: load or generate embeddings
    embeddings_path = os.path.join(os.path.dirname(csv_path), 'embeddings.npz')
    if os.path.exists(embeddings_path):
        print(f"Loading embeddings from {embeddings_path}...")
        job_embeddings, _, _ = load_embeddings(embeddings_path)
        print(f"Loaded embeddings: shape {job_embeddings.shape}")
    else:
        print(f"Generating embeddings for {len(df)} jobs...")
        embeddings = []
        for idx, text in enumerate(df['processed_text']):
            if idx % 100 == 0:
                print(f"  Progress: {idx}/{len(df)}")
            embedding = generate_embedding(text, tokenizer, model, device)
            embeddings.append(embedding)
        job_embeddings = np.array(embeddings)
        print(f"Embeddings generated: shape {job_embeddings.shape}")
        # Save embeddings for faster restarts
        try:
            save_embeddings(embeddings_path, job_embeddings, df)
            print(f"Saved embeddings to {embeddings_path}")
        except Exception as e:
            print(f"Warning: failed to save embeddings: {e}")
    
    # Display job categories/types available
    print("\n" + "="*70)
    print("JOB SEARCH APPLICATION - BERT Semantic Search")
    print("="*70)
    
    if 'Job Title' in df.columns:
        sample_titles = df['Job Title'].dropna().head(10).tolist()
        print("\nSample job titles in dataset:")
        for title in sample_titles:
            print(f"  - {title}")
    
    print("\nYou can search for jobs using natural language queries.")
    print("Examples:")
    print("  - 'machine learning engineer with python'")
    print("  - 'data scientist with SQL experience'")
    print("  - 'software developer backend API'")
    print("  - 'project manager agile scrum'")
    print("\n" + "="*70)
    
    # Interactive search loop with optional filtering
    print("\nYou can filter by location or remote/on-site by including 'location:<city>' or 'remote:true' in your query.")
    while True:
        query = input("\nEnter your search query (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Exiting. Goodbye!")
            break
        
        if not query:
            print("Please enter a valid query.")
            continue
        
        # Extract filters from query (simple syntax: location:city, remote:true/false)
        filters = {}
        parts = [p.strip() for p in query.split()]
        query_terms = []
        for p in parts:
            if p.startswith('location:'):
                filters['location'] = p.split(':', 1)[1]
            elif p.startswith('remote:'):
                filters['remote'] = p.split(':', 1)[1].lower() in ['true', '1', 'yes']
            else:
                query_terms.append(p)
        search_text = ' '.join(query_terms)

        # Perform semantic search
        print(f"\nSearching for: '{search_text}' with filters: {filters}")
        results = semantic_search(search_text, job_embeddings, df, tokenizer, model, device, top_k=20)

        # Apply filters to results
        if filters:
            mask = np.ones(len(results), dtype=bool)
            if 'location' in filters and 'Location' in df.columns:
                mask = mask & results['Location'].str.contains(filters['location'], case=False, na=False).values
            if 'remote' in filters and 'Remote' in df.columns:
                # assume Remote column contains booleans or strings
                if results['Remote'].dtype == bool:
                    mask = mask & (results['Remote'].values == filters['remote'])
                else:
                    mask = mask & results['Remote'].astype(str).str.lower().isin(['true', 'yes']) if filters['remote'] else mask
            results = results[mask]
        # Keep top 5 after filtering
        results = results.head(5)
        
        # Display results
        print(f"\nTop 5 matching jobs:\n")
        for idx, row in results.iterrows():
            print(f"--- Match #{results.index.get_loc(idx) + 1} (Similarity: {row['similarity_score']:.4f}) ---")
            
            if 'Job Title' in row:
                print(f"Title: {row['Job Title']}")
            elif 'title' in row:
                print(f"Title: {row['title']}")
            
            if 'Job Description' in row:
                desc = row['Job Description']
                if isinstance(desc, str) and len(desc) > 200:
                    desc = desc[:200] + "..."
                print(f"Description: {desc}")
            elif 'description' in row:
                desc = row['description']
                if isinstance(desc, str) and len(desc) > 200:
                    desc = desc[:200] + "..."
                print(f"Description: {desc}")
            
            print()


if __name__ == "__main__":
    main()
