"""
Multi-modal embeddings with pretrained CLIP model (ViT-B/32)

This module generates embeddings for images and their text descriptions using CLIP,
and provides semantic search functionality:
- Search images by text query
- Search text descriptions by image query

Data paths:
  images folder: /Users/madhukarraj/Downloads/cbe-vector-databases-main/data/images
  CSV file: /Users/madhukarraj/Downloads/cbe-vector-databases-main/data/image_descriptions.csv
"""

import os
from typing import List, Tuple, Optional

import clip
import torch
import pandas as pd
from PIL import Image

# IPython display imports (for Jupyter notebooks)
try:
    from IPython.display import display, HTML
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False


# Configuration
MODEL_NAME = "ViT-B/32"
IMAGES_FOLDER = "/Users/madhukarraj/Downloads/cbe-vector-databases-main/data/images"
CSV_PATH = "/Users/madhukarraj/Downloads/cbe-vector-databases-main/data/image_descriptions.csv"
EMBEDDINGS_PATH = "/Users/madhukarraj/vscode/g-ai/search-app/clip_embeddings.pt"


def get_device() -> torch.device:
    """Return the torch device to use (cuda if available, else mps for Apple Silicon, else cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_name: str = MODEL_NAME, device: torch.device = None):
    """Load the CLIP model and preprocessing function.

    Args:
        model_name: CLIP model variant (default: ViT-B/32)
        device: Torch device to use

    Returns:
        Tuple of (model, preprocess_fn, device)
    """
    if device is None:
        device = get_device()

    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    print(f"Loaded CLIP model: {model_name}")
    print(f"Device: {device}")

    return model, preprocess, device


def load_dataset(
    images_folder: str = IMAGES_FOLDER,
    csv_path: str = CSV_PATH
) -> Tuple[List[str], List[str]]:
    """Load images and their descriptions from CSV.

    The CSV should have columns: 'Image ID', 'Description'
    Image files should be named with the ID as prefix (e.g., '1_apple.png')

    Args:
        images_folder: Path to folder containing images
        csv_path: Path to CSV file with descriptions

    Returns:
        Tuple of (image_paths, descriptions)
    """
    df = pd.read_csv(csv_path)

    # Get list of image files
    image_files = os.listdir(images_folder)

    image_paths = []
    descriptions = []

    for _, row in df.iterrows():
        image_id = str(row["Image ID"])
        description = row["Description"]

        # Find matching image file (starts with ID followed by underscore or dot)
        matching_files = [
            f for f in image_files
            if f.startswith(f"{image_id}_") or f.startswith(f"{image_id}.")
        ]

        if matching_files:
            image_path = os.path.join(images_folder, matching_files[0])
            image_paths.append(image_path)
            descriptions.append(description)
        else:
            print(f"Warning: No image found for ID {image_id}")

    print(f"Loaded {len(image_paths)} image-description pairs")
    return image_paths, descriptions


def preprocess_data(
    model,
    preprocess,
    image_paths: List[str],
    descriptions: List[str],
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Preprocess images and text descriptions for CLIP.

    Args:
        model: CLIP model (unused but kept for API consistency)
        preprocess: CLIP preprocessing function for images
        image_paths: List of image file paths
        descriptions: List of text descriptions
        device: Torch device

    Returns:
        Tuple of (image_tensors, text_tokens) ready for encoding
    """
    # Preprocess images
    images = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        image_tensor = preprocess(image)
        images.append(image_tensor)

    image_tensors = torch.stack(images).to(device)

    # Tokenize text (truncate=True handles text > 77 tokens)
    text_tokens = clip.tokenize(descriptions, truncate=True).to(device)

    print(f"Preprocessed {len(images)} images and {len(descriptions)} descriptions")
    return image_tensors, text_tokens


def generate_embeddings(
    model,
    image_tensors: torch.Tensor,
    text_tokens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate normalized embeddings for images and text.

    Args:
        model: CLIP model
        image_tensors: Preprocessed image tensors
        text_tokens: Tokenized text

    Returns:
        Tuple of (image_embeddings, text_embeddings) normalized to unit length
    """
    with torch.no_grad():
        # Generate embeddings
        image_embeddings = model.encode_image(image_tensors)
        text_embeddings = model.encode_text(text_tokens)

        # Normalize to unit length for cosine similarity
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    print(f"Generated embeddings - Images: {image_embeddings.shape}, Text: {text_embeddings.shape}")
    return image_embeddings, text_embeddings


def save_embeddings(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    image_paths: List[str],
    descriptions: List[str],
    save_path: str = EMBEDDINGS_PATH
):
    """Save embeddings and metadata to a file.

    Args:
        image_embeddings: Image embedding tensor
        text_embeddings: Text embedding tensor
        image_paths: List of image paths
        descriptions: List of descriptions
        save_path: Path to save the embeddings
    """
    torch.save({
        "image_embeddings": image_embeddings.cpu(),
        "text_embeddings": text_embeddings.cpu(),
        "image_paths": image_paths,
        "descriptions": descriptions
    }, save_path)

    print(f"Saved embeddings to {save_path}")


def load_embeddings(load_path: str = EMBEDDINGS_PATH) -> dict:
    """Load saved embeddings and metadata.

    Args:
        load_path: Path to the saved embeddings file

    Returns:
        Dictionary with keys: image_embeddings, text_embeddings, image_paths, descriptions
    """
    data = torch.load(load_path, weights_only=False)
    print(f"Loaded embeddings from {load_path}")
    return data


# ============ Display Functions (IPython/Jupyter) ============

def display_image(image_path: str, width: int = 200):
    """Display an image in IPython/Jupyter notebook.

    Args:
        image_path: Path to the image file
        width: Display width in pixels
    """
    if IPYTHON_AVAILABLE:
        img = Image.open(image_path).convert("RGB")
        # Resize for display
        aspect_ratio = img.height / img.width
        new_height = int(width * aspect_ratio)
        img_resized = img.resize((width, new_height))
        display(img_resized)
    else:
        print(f"[Image: {os.path.basename(image_path)}]")


def display_search_results(results: List[dict], show_images: bool = True, image_width: int = 150):
    """Display search results with images in IPython/Jupyter.

    Args:
        results: List of result dicts with 'path', 'description', 'score' keys
        show_images: Whether to display images
        image_width: Width for displayed images
    """
    if IPYTHON_AVAILABLE and show_images:
        html_parts = ['<div style="display: flex; flex-wrap: wrap; gap: 20px;">']

        for i, r in enumerate(results, 1):
            score = r['score']
            desc = r['description'][:100] + "..." if len(r['description']) > 100 else r['description']
            img_name = os.path.basename(r['path'])

            html_parts.append(f'''
            <div style="border: 1px solid #ddd; padding: 10px; border-radius: 8px; width: {image_width + 20}px;">
                <img src="file://{r['path']}" width="{image_width}" style="border-radius: 4px;">
                <p style="margin: 5px 0; font-weight: bold;">#{i} {img_name}</p>
                <p style="margin: 5px 0; color: green;">Score: {score:.4f}</p>
                <p style="margin: 5px 0; font-size: 12px; color: #666;">{desc}</p>
            </div>
            ''')

        html_parts.append('</div>')
        display(HTML(''.join(html_parts)))
    else:
        for i, r in enumerate(results, 1):
            print(f"  {i}. {os.path.basename(r['path'])} (score: {r['score']:.4f})")
            print(f"     {r['description'][:80]}...")


def display_image_with_results(query_image: str, results: List[dict], image_width: int = 150):
    """Display query image alongside search results in IPython/Jupyter.

    Args:
        query_image: Path to the query image
        results: List of result dicts
        image_width: Width for displayed images
    """
    if IPYTHON_AVAILABLE:
        html_parts = ['<div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start;">']

        # Query image
        html_parts.append(f'''
        <div style="border: 2px solid #007bff; padding: 10px; border-radius: 8px; width: {image_width + 20}px;">
            <img src="file://{query_image}" width="{image_width}" style="border-radius: 4px;">
            <p style="margin: 5px 0; font-weight: bold; color: #007bff;">Query Image</p>
            <p style="margin: 5px 0; font-size: 12px;">{os.path.basename(query_image)}</p>
        </div>
        ''')

        # Arrow
        html_parts.append('<div style="display: flex; align-items: center; font-size: 24px; padding: 0 10px;">→</div>')

        # Results
        for i, r in enumerate(results, 1):
            score = r['score']
            desc = r['description'][:80] + "..." if len(r['description']) > 80 else r['description']

            html_parts.append(f'''
            <div style="border: 1px solid #ddd; padding: 10px; border-radius: 8px; width: {image_width + 20}px;">
                <img src="file://{r['path']}" width="{image_width}" style="border-radius: 4px;">
                <p style="margin: 5px 0; font-weight: bold;">#{i}</p>
                <p style="margin: 5px 0; color: green;">Score: {score:.4f}</p>
                <p style="margin: 5px 0; font-size: 11px; color: #666;">{desc}</p>
            </div>
            ''')

        html_parts.append('</div>')
        display(HTML(''.join(html_parts)))
    else:
        print(f"Query image: {os.path.basename(query_image)}")
        for i, r in enumerate(results, 1):
            print(f"  {i}. Score: {r['score']:.4f}")
            print(f"     {r['description'][:100]}...")


# ============ Semantic Search Functions ============

def search_image_by_text(
    query: str,
    model=None,
    preprocess=None,
    device: torch.device = None,
    embeddings_path: str = EMBEDDINGS_PATH,
    top_k: int = 5
) -> List[dict]:
    """Search for images that match a text query.

    Args:
        query: Text description to search for
        model: CLIP model (will load if None)
        preprocess: CLIP preprocessing function (unused for text, but kept for API consistency)
        device: Torch device
        embeddings_path: Path to saved embeddings
        top_k: Number of results to return

    Returns:
        List of dicts: [{'path': <path>, 'description': <desc>, 'score': <similarity>}, ...]
    """
    if device is None:
        device = get_device()

    if model is None:
        model, preprocess, device = load_model(device=device)

    # Load saved embeddings
    data = load_embeddings(embeddings_path)
    image_embeddings = data["image_embeddings"].to(device)
    image_paths = data["image_paths"]
    descriptions = data["descriptions"]

    # Encode query text
    query_tokens = clip.tokenize([query], truncate=True).to(device)

    with torch.no_grad():
        query_embedding = model.encode_text(query_tokens)
        query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)

    # Compute cosine similarities
    similarities = (query_embedding @ image_embeddings.T).squeeze(0)

    # Get top-k results
    top_indices = similarities.argsort(descending=True)[:top_k]

    results = []
    for idx in top_indices:
        idx = idx.item()
        results.append({
            "path": image_paths[idx],
            "description": descriptions[idx],
            "score": float(similarities[idx])
        })

    return results


def search_text_by_image(
    query_image: str,
    model=None,
    preprocess=None,
    device: torch.device = None,
    embeddings_path: str = EMBEDDINGS_PATH,
    top_k: int = 5
) -> List[dict]:
    """Search for text descriptions that match an image query.

    Args:
        query_image: Path to the query image
        model: CLIP model (will load if None)
        preprocess: CLIP preprocessing function
        device: Torch device
        embeddings_path: Path to saved embeddings
        top_k: Number of results to return

    Returns:
        List of dicts: [{'path': <path>, 'description': <desc>, 'score': <similarity>}, ...]
    """
    if device is None:
        device = get_device()

    if model is None:
        model, preprocess, device = load_model(device=device)

    # Load saved embeddings
    data = load_embeddings(embeddings_path)
    text_embeddings = data["text_embeddings"].to(device)
    image_paths = data["image_paths"]
    descriptions = data["descriptions"]

    # Preprocess and encode query image
    image = Image.open(query_image).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        query_embedding = model.encode_image(image_tensor)
        query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)

    # Compute cosine similarities
    similarities = (query_embedding @ text_embeddings.T).squeeze(0)

    # Get top-k results
    top_indices = similarities.argsort(descending=True)[:top_k]

    results = []
    for idx in top_indices:
        idx = idx.item()
        results.append({
            "path": image_paths[idx],
            "description": descriptions[idx],
            "score": float(similarities[idx])
        })

    return results


# ============ Main Pipeline ============

def generate_and_save_embeddings():
    """Main pipeline to generate and save embeddings for the dataset."""
    # Load model
    model, preprocess, device = load_model()

    # Load dataset
    image_paths, descriptions = load_dataset()

    # Preprocess data
    image_tensors, text_tokens = preprocess_data(
        model, preprocess, image_paths, descriptions, device
    )

    # Generate embeddings
    image_embeddings, text_embeddings = generate_embeddings(
        model, image_tensors, text_tokens
    )

    # Save embeddings
    save_embeddings(
        image_embeddings, text_embeddings,
        image_paths, descriptions
    )

    return image_embeddings, text_embeddings, image_paths, descriptions


def demo_search(show_images: bool = True):
    """Demonstrate search functionality.

    Args:
        show_images: If True and running in IPython/Jupyter, display images inline
    """
    model, preprocess, device = load_model()

    print("\n" + "=" * 60)
    print("DEMO: Search Image by Text")
    print("=" * 60)

    text_queries = [
        "a red apple",
        "yellow fruit",
        "vegetable basket"
    ]

    for query in text_queries:
        print(f"\nQuery: '{query}'")
        results = search_image_by_text(query, model, preprocess, device, top_k=3)
        display_search_results(results, show_images=show_images)

    print("\n" + "=" * 60)
    print("DEMO: Search Text by Image")
    print("=" * 60)

    # Use first image from dataset as query
    data = load_embeddings()
    query_image = data["image_paths"][0]

    print(f"\nQuery image: {os.path.basename(query_image)}")
    results = search_text_by_image(query_image, model, preprocess, device, top_k=3)
    display_image_with_results(query_image, results)


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]

    if "--demo" in args:
        # Run demo (assumes embeddings already generated)
        demo_search(show_images=IPYTHON_AVAILABLE)
    else:
        # Generate embeddings and run demo
        print("Generating embeddings...")
        generate_and_save_embeddings()

        print("\nRunning search demo...")
        demo_search(show_images=IPYTHON_AVAILABLE)


# ============ Convenience Functions for Jupyter ============

def interactive_text_search(query: str, top_k: int = 5):
    """Search images by text query with visual display (for Jupyter).

    Args:
        query: Text description to search for
        top_k: Number of results to return
    """
    print(f"Searching for: '{query}'")
    model, preprocess, device = load_model()
    results = search_image_by_text(query, model, preprocess, device, top_k=top_k)
    display_search_results(results, show_images=True)
    return results


def interactive_image_search(image_path: str, top_k: int = 5):
    """Search text descriptions by image with visual display (for Jupyter).

    Args:
        image_path: Path to query image
        top_k: Number of results to return
    """
    print(f"Searching with image: {os.path.basename(image_path)}")
    model, preprocess, device = load_model()
    results = search_text_by_image(image_path, model, preprocess, device, top_k=top_k)
    display_image_with_results(image_path, results)
    return results
