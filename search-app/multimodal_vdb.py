"""
Multimodal Embeddings with ChromaDB and OpenCLIP

This module provides functionality for creating and querying a multimodal
vector database using ChromaDB with OpenCLIP (ViT-B/32) embeddings.
Supports both text and image queries against a collection of images with descriptions.
"""

import os
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import pandas as pd


# Default persistent storage path
DEFAULT_PERSIST_DIR = os.path.join(os.path.dirname(__file__), ".chromadb")


class MultimodalVectorDB:
    """A multimodal vector database using ChromaDB and OpenCLIP embeddings."""

    def __init__(
        self,
        collection_name: str = "multimodal_embeddings_collection",
        persist_directory: str = None,
    ):
        """
        Initialize the multimodal vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Path to persist ChromaDB data (default: .chromadb in script dir)
        """
        # Configure embedding model (ViT-B-32 CLIP variant)
        self.embedding_function = OpenCLIPEmbeddingFunction(model_name="ViT-B-32")
        self.data_loader = ImageLoader()

        # Use persistent client to store data on disk
        self.persist_directory = persist_directory or DEFAULT_PERSIST_DIR
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # Create collection with cosine similarity
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
            data_loader=self.data_loader,
        )
        self.collection_name = collection_name

    def is_loaded(self) -> bool:
        """Check if data has already been loaded into the collection."""
        return self.collection.count() > 0

    def load_data(self, csv_path: str, images_folder: str, force_reload: bool = False) -> None:
        """
        Load images and descriptions from CSV and image folder.

        Args:
            csv_path: Path to CSV file with Image ID and Description columns
            images_folder: Path to folder containing images
            force_reload: If True, clear existing data and reload
        """
        # Skip if data already loaded (unless force_reload)
        if self.is_loaded() and not force_reload:
            print(f"Data already loaded ({self.collection.count()} items). Skipping load.")
            return

        # Clear existing data if force reloading
        if force_reload and self.is_loaded():
            print("Force reload: clearing existing data...")
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"},
                data_loader=self.data_loader,
            )

        # Load descriptions from CSV
        df = pd.read_csv(csv_path)

        # Get list of image files
        image_files = os.listdir(images_folder)

        # Track data for adding to collection
        image_paths = []
        image_ids = []
        descriptions = []
        description_ids = []

        # Match images with descriptions
        for _, row in df.iterrows():
            img_id = row["Image ID"]
            desc = row["Description"]

            # Find matching image file (starts with image ID)
            matching_files = [f for f in image_files if f.startswith(f"{img_id}_")]

            if matching_files:
                img_path = os.path.join(images_folder, matching_files[0])
                image_paths.append(img_path)
                image_ids.append(f"img_{img_id}")
                descriptions.append(desc)
                description_ids.append(f"desc_{img_id}")

        print(f"Found {len(image_paths)} image-description pairs")

        # Add data to collection
        self._add_to_collection(image_ids, image_paths, descriptions, description_ids)

    def _add_to_collection(
        self,
        image_ids: list,
        image_paths: list,
        descriptions: list,
        description_ids: list,
    ) -> None:
        """Add images and descriptions to the ChromaDB collection."""
        for img_id, img_path, desc, desc_id in zip(
            image_ids, image_paths, descriptions, description_ids
        ):
            # Add image to collection
            self.collection.add(
                ids=[img_id],
                uris=[img_path],
                metadatas=[{"image_uri": img_path, "description": desc}],
            )

            # Add description to collection
            self.collection.add(
                ids=[desc_id],
                documents=[desc],
                metadatas=[{"image_uri": img_path, "description": desc}],
            )

        print(f"Added {len(image_ids)} images and {len(descriptions)} descriptions")

    def query_by_text(self, query_text: str, n_results: int = 5) -> dict:
        """
        Query the collection using text.

        Args:
            query_text: Text query to search for
            n_results: Number of results to return

        Returns:
            Query results from ChromaDB
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )
        return results

    def query_by_image(self, image_path: str, n_results: int = 5) -> dict:
        """
        Query the collection using an image.

        Args:
            image_path: Path to query image
            n_results: Number of results to return

        Returns:
            Query results from ChromaDB
        """
        results = self.collection.query(
            query_uris=[image_path],
            n_results=n_results,
        )
        return results

    def get_collection_count(self) -> int:
        """Return the number of items in the collection."""
        return self.collection.count()


def print_results(results: dict, query_type: str = "text") -> None:
    """Pretty print query results."""
    print(f"\n{'='*60}")
    print(f"Query Type: {query_type}")
    print(f"{'='*60}")

    if results["metadatas"] and results["metadatas"][0]:
        for i, metadata in enumerate(results["metadatas"][0]):
            distance = results["distances"][0][i] if results["distances"] else "N/A"
            print(f"\nResult {i+1}:")
            print(f"  Distance: {distance:.4f}" if isinstance(distance, float) else f"  Distance: {distance}")
            print(f"  Image: {metadata.get('image_uri', 'N/A')}")
            print(f"  Description: {metadata.get('description', 'N/A')[:100]}...")
    else:
        print("No results found.")


def main():
    """Main function demonstrating multimodal vector database usage."""
    # Paths to data
    csv_path = "/Users/madhukarraj/Downloads/cbe-vector-databases-main/data/image_descriptions.csv"
    images_folder = "/Users/madhukarraj/Downloads/cbe-vector-databases-main/data/images"

    # Check if data exists
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
    if not os.path.exists(images_folder):
        print(f"Error: Images folder not found at {images_folder}")
        return

    # Initialize vector database (persistent storage)
    print("Initializing multimodal vector database...")
    vdb = MultimodalVectorDB()
    print(f"Persistent storage: {vdb.persist_directory}")

    # Load data (skips if already loaded)
    print("\nLoading data...")
    vdb.load_data(csv_path, images_folder)
    print(f"Collection count: {vdb.get_collection_count()}")

    # Text query examples
    text_queries = [
        "red fruit",
        "yellow vegetable",
        "citrus fruit with vitamin C",
    ]

    for query in text_queries:
        print(f"\n>>> Text Query: '{query}'")
        results = vdb.query_by_text(query, n_results=3)
        print_results(results, "text")

    # Image query example (using first image in folder)
    image_files = os.listdir(images_folder)
    if image_files:
        query_image = os.path.join(images_folder, image_files[0])
        print(f"\n>>> Image Query: '{query_image}'")
        results = vdb.query_by_image(query_image, n_results=3)
        print_results(results, "image")


if __name__ == "__main__":
    main()
