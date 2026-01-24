from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from uuid import uuid4
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables from .env file
load_dotenv()


def initialize_chroma_store(persist_directory='./.chromadb'):
    """
    Initialize ChromaDB vector store with OpenAI embeddings.

    Args:
        persist_directory (str): Directory path to persist the vector store.
                                Defaults to './.chromadb'.

    Returns:
        Chroma: Initialized ChromaDB vector store instance.

    Raises:
        Exception: If initialization fails (e.g., missing API key, connection issues).
    """
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory
        )

        # Check if collection is empty and add sample documents if needed
        if vector_store._collection.count() == 0:
            add_sample_documents(vector_store)
            print("ChromaDB initialized with sample documents")

        return vector_store
    except Exception as e:
        raise Exception(f"Failed to initialize ChromaDB: {str(e)}")


def search_documents(vector_store, query, top_k=3):
    """
    Perform semantic similarity search on the vector store.

    Args:
        vector_store (Chroma): The ChromaDB vector store instance.
        query (str): The search query text.
        top_k (int): Number of top results to return. Defaults to 3.

    Returns:
        List[Dict[str, Any]]: List of search results, each containing:
            - 'content': The document content
            - 'metadata': Document metadata

    Raises:
        ValueError: If query is empty or top_k is invalid.
        Exception: If search operation fails.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    try:
        results = vector_store.similarity_search(query, k=top_k)
        return [
            {
                'content': doc.page_content,
                'metadata': doc.metadata
            }
            for doc in results
        ]
    except Exception as e:
        raise Exception(f"Search failed: {str(e)}")


def add_sample_documents(vector_store):
    """
    Add sample documents to the vector store for testing purposes.

    Args:
        vector_store (Chroma): The ChromaDB vector store instance.

    Returns:
        None
    """
    document_1 = Document(
        page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
        metadata={"source": "tweet"},
    )

    document_2 = Document(
        page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
        metadata={"source": "news"},
    )

    document_3 = Document(
        page_content="Building an exciting new project with LangChain - come check it out!",
        metadata={"source": "tweet"},
    )

    documents = [document_1, document_2, document_3]
    uuids = [str(uuid4()) for _ in range(len(documents))]

    vector_store.add_documents(documents=documents, ids=uuids)
    print(f"Added {len(documents)} sample documents to the vector store")


# Standalone script execution for testing
if __name__ == '__main__':
    # Initialize vector store
    vector_store = initialize_chroma_store()

    # Test query
    query = "What's the weather going to be like tomorrow?"
    results = search_documents(vector_store, query, top_k=1)

    print(f"\nQuery: {query}")
    print(f"Results: {results}")