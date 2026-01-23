"""
Video embedding + semantic search utilities

Step 1: Imports and necessary libraries

This module will be built in steps:
- Step 1 (this file): imports, constants and helper utilities
- Step 2: load pretrained R3D-18 model
- Step 3: implement preprocess_video()
- Step 4: implement generate_video_embedding()
- Step 5: implement semantic_search()

Embedding choices (provided by user):
  embedding_type = 'video'
  embedding_model = 'r3d_18'  # 3D ResNet-18 for video
"""

import os
from typing import List

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

# torchvision imports for video models and transforms
import torchvision.transforms as transforms
import torchvision.models.video as video_models

# For video reading - we'll use torchvision's video reader or opencv
try:
    import cv2
    VIDEO_BACKEND = 'opencv'
except ImportError:
    VIDEO_BACKEND = 'torchvision'


# User-selected configuration
EMBEDDING_TYPE = 'video'
EMBEDDING_MODEL_NAME = 'r3d_18'


def is_video_file(path: str) -> bool:
    """Return True if path has a common video file extension."""
    video_ext = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    _, ext = os.path.splitext(path.lower())
    return ext in video_ext


def list_video_files(folder: str) -> List[str]:
    """List video files inside a folder (non-recursive).

    Returns absolute paths.
    """
    if not os.path.isdir(folder):
        return []
    files = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and is_video_file(path):
            files.append(path)
    return files


def get_device() -> torch.device:
    """Return the torch device to use (cuda if available)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ensure_folder(folder: str):
    """Create folder if it doesn't exist."""
    os.makedirs(folder, exist_ok=True)


def load_video_model(model_name: str = EMBEDDING_MODEL_NAME, device: torch.device = None):
    """Load a pretrained 3D CNN video model and strip its final classification layer.

    Returns the model in evaluation mode on the desired device.
    """
    if device is None:
        device = get_device()

    model_name = model_name.lower()
    if model_name == 'r3d_18' or model_name.startswith('r3d'):
        # Load pretrained R3D-18 (3D ResNet-18 for video)
        try:
            model = video_models.r3d_18(pretrained=True)
        except TypeError:
            # Fallback for newer torchvision versions
            from torchvision.models.video import R3D_18_Weights
            model = video_models.r3d_18(weights=R3D_18_Weights.DEFAULT)

        # Replace final fully-connected layer with identity so output is embedding vector
        model.fc = torch.nn.Identity()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.to(device)
    model.eval()
    return model, device


def preprocess_video(video_path: str, device: torch.device = None, num_frames: int = 16, size: int = 112) -> torch.Tensor:
    """Load a video from `video_path` and return a preprocessed tensor ready for the R3D model.

    Steps:
    - Open video and extract frames
    - Sample `num_frames` frames uniformly from the video
    - Resize frames to `size` x `size`
    - Convert to tensor and normalize using Kinetics mean/std
    - Create shape (1, 3, num_frames, size, size) - [batch, channels, temporal, height, width]

    Returns a tensor with shape (1, 3, num_frames, size, size) on the requested device.
    """
    if device is None:
        device = get_device()

    # Read video frames using opencv
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames extracted from video: {video_path}")

    # Sample num_frames uniformly
    indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    sampled_frames = [frames[i] for i in indices]

    # Resize and convert to tensor
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
    ])

    tensors = [transform(frame) for frame in sampled_frames]
    
    # Stack frames: (num_frames, C, H, W) -> (C, num_frames, H, W)
    video_tensor = torch.stack(tensors, dim=1)  # (C, T, H, W)
    
    # Add batch dimension: (1, C, T, H, W)
    video_tensor = video_tensor.unsqueeze(0).to(device)
    
    return video_tensor


def generate_video_embedding(model: torch.nn.Module, video_tensor: torch.Tensor) -> np.ndarray:
    """Run `video_tensor` (1,C,T,H,W) through `model` and return a 1-D numpy embedding.

    The `model` is expected to have its final classification layer replaced by Identity
    (so its output is a feature vector). The function will detach and move the result
    to CPU before converting to numpy.
    """
    if video_tensor.dim() == 4:
        # If missing batch dimension, add it
        video_tensor = video_tensor.unsqueeze(0)

    with torch.no_grad():
        feats = model(video_tensor)

    if isinstance(feats, tuple):
        feats = feats[0]

    feats = feats.detach().cpu().squeeze()
    return feats.numpy()


def semantic_search(folder: str, query_video_path: str, model: torch.nn.Module = None, device: torch.device = None,
                    top_k: int = 5, cache_path: str = None) -> List[dict]:
    """Compute semantic search over videos in `folder` using `model`.

    - If `cache_path` is provided and exists, load embeddings from it (`npz` with arrays 'paths' and 'embs').
    - Otherwise compute embeddings for all videos in `folder` and optionally save to `cache_path`.
    - Compute embedding for `query_video_path` and return `top_k` matches sorted by cosine similarity.

    Returns a list of dicts: [{'path': <path>, 'score': <similarity>}, ...]
    """
    if device is None:
        device = get_device()

    if model is None:
        model, device = load_video_model(device=device)

    # Load or compute gallery embeddings
    video_paths = list_video_files(folder)
    if len(video_paths) == 0:
        return []

    gallery_embs = None
    gallery_paths = video_paths

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
                t = preprocess_video(p, device=device)
                e = generate_video_embedding(model, t)
                emb_list.append(e)
                print(f"Processed: {os.path.basename(p)}")
            except Exception as ex:
                print(f"Error processing {p}: {ex}")
                emb_list.append(np.zeros(512, dtype=np.float32))

        gallery_embs = np.vstack([e.reshape(1, -1) for e in emb_list])
        if cache_path:
            try:
                np.savez_compressed(cache_path, paths=np.array(gallery_paths), embs=gallery_embs)
                print(f"Saved embeddings cache to {cache_path}")
            except Exception as ex:
                print(f"Could not save cache: {ex}")

    # Query embedding
    q_t = preprocess_video(query_video_path, device=device)
    q_e = generate_video_embedding(model, q_t).reshape(1, -1)

    # Compute cosine similarities
    sims = cosine_similarity(q_e, gallery_embs)[0]
    idx = np.argsort(-sims)[:top_k]

    results = []
    for i in idx:
        results.append({'path': gallery_paths[i], 'score': float(sims[i])})

    return results


if __name__ == '__main__':
    print('video_search module loaded')
    print('Embedding type:', EMBEDDING_TYPE)
    print('Embedding model:', EMBEDDING_MODEL_NAME)
    print('Video backend:', VIDEO_BACKEND)
