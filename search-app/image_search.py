"""
Image embedding + semantic search utilities

Step 1: imports and helpers

This module will be built in steps:
- Step 1 (this file): imports, constants and helper utilities
- Step 2: load pretrained model (remove final layer)
- Step 3: implement preprocess_image()
- Step 4: implement generate_image_embedding()
- Step 5: implement semantic_search()

Embedding choices (provided by user):
  embedding_type = 'image'
  embedding_model = 'resnet18'  # mapped from 'ResNet-18'

After this step, I will wait for your approval before loading the model.
"""

import os
from typing import List

import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# torchvision imports will be used in the next step to load model and transforms
import torchvision.transforms as transforms
import torchvision.models as models


# User-selected configuration
EMBEDDING_TYPE = 'image'
EMBEDDING_MODEL_NAME = 'resnet18'


def is_image_file(path: str) -> bool:
    """Return True if path has a common image file extension."""
    img_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    _, ext = os.path.splitext(path.lower())
    return ext in img_ext


def list_image_files(folder: str) -> List[str]:
    """List image files inside a folder (non-recursive).

    Returns absolute paths.
    """
    if not os.path.isdir(folder):
        return []
    files = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and is_image_file(path):
            files.append(path)
    return files


def get_device() -> torch.device:
    """Return the torch device to use (cuda if available)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ensure_folder(folder: str):
    os.makedirs(folder, exist_ok=True)


if __name__ == '__main__':
    print('image_search module loaded')
    print('Embedding type:', EMBEDDING_TYPE)
    print('Embedding model:', EMBEDDING_MODEL_NAME)


def load_image_model(model_name: str = EMBEDDING_MODEL_NAME, device: torch.device = None):
    """Load a pretrained CNN model and strip its final classification layer.

    Returns the model in evaluation mode on the desired device.
    """
    if device is None:
        device = get_device()

    model_name = model_name.lower()
    if model_name == 'resnet18' or model_name.startswith('resnet'):
        # Load pretrained ResNet-18
        try:
            model = models.resnet18(pretrained=True)
        except TypeError:
            # Fallback for newer torchvision versions
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Replace final fully-connected layer with identity so output is embedding vector
        model.fc = torch.nn.Identity()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.to(device)
    model.eval()
    return model, device


def preprocess_image(image_path: str, device: torch.device = None, size: int = 224) -> torch.Tensor:
    """Load an image from `image_path` and return a preprocessed tensor ready for the model.

    Steps:
    - Open image and convert to RGB
    - Resize shorter side to 256 and center-crop to `size` x `size`
    - Convert to tensor and normalize using ImageNet mean/std
    - Add a batch dimension and move to `device`

    Returns a tensor with shape (1, 3, size, size) on the requested device.
    """
    if device is None:
        device = get_device()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    return tensor


def generate_image_embedding(model: torch.nn.Module, image_tensor: torch.Tensor) -> np.ndarray:
    """Run `image_tensor` (1,3,H,W) through `model` and return a 1-D numpy embedding.

    The `model` is expected to have its final classification layer replaced by Identity
    (so its output is a feature vector). The function will detach and move the result
    to CPU before converting to numpy.
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        feats = model(image_tensor)

    if isinstance(feats, tuple):
        feats = feats[0]

    feats = feats.detach().cpu().squeeze()
    return feats.numpy()


def _smoke_test_embedding(sample_path: str = None):
    """Quick smoke test: load model, preprocess a tiny sample image (or generate one), and print embedding shape."""
    model, device = load_image_model()
    # create a small sample image if none provided
    if sample_path is None:
        sample_path = 'test_sample_image.jpg'
        if not os.path.exists(sample_path):
            img = Image.new('RGB', (300, 300), color=(128, 128, 128))
            img.save(sample_path)

    tensor = preprocess_image(sample_path, device=device)
    emb = generate_image_embedding(model, tensor)
    print('Sample embedding shape:', emb.shape)
    return emb


def semantic_search(folder: str, query_image_path: str, model: torch.nn.Module = None, device: torch.device = None,
                    top_k: int = 5, cache_path: str = None) -> List[dict]:
    """Compute semantic search over images in `folder` using `model`.

    - If `cache_path` is provided and exists, load embeddings from it (`npz` with arrays 'paths' and 'embs').
    - Otherwise compute embeddings for all images in `folder` and optionally save to `cache_path`.
    - Compute embedding for `query_image_path` and return `top_k` matches sorted by cosine similarity.

    Returns a list of dicts: [{'path': <path>, 'score': <similarity>}, ...]
    """
    if device is None:
        device = get_device()

    if model is None:
        model, device = load_image_model(device=device)

    # Load or compute gallery embeddings
    img_paths = list_image_files(folder)
    if len(img_paths) == 0:
        return []

    gallery_embs = None
    gallery_paths = img_paths

    if cache_path and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            gallery_paths = data['paths'].tolist()
            gallery_embs = data['embs']
        except Exception:
            gallery_embs = None

    if gallery_embs is None:
        emb_list = []
        for p in gallery_paths:
            try:
                t = preprocess_image(p, device=device)
                e = generate_image_embedding(model, t)
                emb_list.append(e)
            except Exception:
                emb_list.append(np.zeros(1, dtype=np.float32))

        gallery_embs = np.vstack([e.reshape(1, -1) for e in emb_list])
        if cache_path:
            try:
                np.savez_compressed(cache_path, paths=np.array(gallery_paths), embs=gallery_embs)
            except Exception:
                pass

    # Query embedding
    q_t = preprocess_image(query_image_path, device=device)
    q_e = generate_image_embedding(model, q_t).reshape(1, -1)

    # Compute cosine similarities
    sims = cosine_similarity(q_e, gallery_embs)[0]
    idx = np.argsort(-sims)[:top_k]

    results = []
    for i in idx:
        results.append({'path': gallery_paths[i], 'score': float(sims[i])})

    return results


def _smoke_test_search():
    """Create sample images, run semantic_search and print results."""
    # Use fixtures directory relative to this file's location
    base = os.path.dirname(os.path.dirname(__file__))
    folder = os.path.join(base, 'fixtures', 'images')
    cache_path = os.path.join(base, 'fixtures', 'cache', 'test_images_cache.npz')
    ensure_folder(folder)
    # create some simple colored images
    colors = [(200, 50, 50), (50, 200, 50), (50, 50, 200), (180, 180, 50), (50, 180, 180)]
    paths = []
    for i, c in enumerate(colors):
        p = os.path.join(folder, f'sample_{i}.jpg')
        if not os.path.exists(p):
            Image.new('RGB', (300, 300), color=c).save(p)
        paths.append(p)

    query = paths[0]
    model, device = load_image_model()
    results = semantic_search(folder, query, model=model, device=device, top_k=3, cache_path=cache_path)
    print('Search results:', results)
    return results
