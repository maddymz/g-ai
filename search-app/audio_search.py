"""
Audio embedding + semantic search utilities using Mel Spectrograms

Step 1: Imports and necessary libraries

This module will be built in steps:
- Step 1 (this file): imports, constants and helper utilities
- Step 2: setup mel spectrogram configuration
- Step 3: implement preprocess_audio()
- Step 4: implement generate_audio_embedding()
- Step 5: implement semantic_search()

Embedding choices (provided by user):
  embedding_type = 'audio'
  embedding_model = 'mel spectrograms'
"""

import os
from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# torchaudio for audio processing and mel spectrogram generation
try:
    import torch
    import torchaudio
    import torchaudio.transforms as T
    AUDIO_BACKEND = 'torchaudio'
except ImportError:
    AUDIO_BACKEND = 'none'
    print("Warning: torchaudio not installed. Install with: pip install torchaudio")


# User-selected configuration
EMBEDDING_TYPE = 'audio'
EMBEDDING_MODEL_NAME = 'mel_spectrograms'

# Mel spectrogram configuration parameters
SAMPLE_RATE = 16000  # Standard sample rate for audio processing
N_FFT = 2048  # FFT window size
HOP_LENGTH = 512  # Number of samples between successive frames
N_MELS = 128  # Number of mel frequency bins
F_MIN = 0.0  # Minimum frequency
F_MAX = 8000.0  # Maximum frequency (typically sample_rate/2 but can be lower)


def get_mel_spectrogram_transform(sample_rate: int = SAMPLE_RATE):
    """Create and return a mel spectrogram transform using torchaudio.

    Returns a MelSpectrogram transform configured with standard parameters.
    """
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=F_MIN,
        f_max=F_MAX
    )
    return mel_transform


def is_audio_file(path: str) -> bool:
    """Return True if path has a common audio file extension."""
    audio_ext = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
    _, ext = os.path.splitext(path.lower())
    return ext in audio_ext


def list_audio_files(folder: str) -> List[str]:
    """List audio files inside a folder (non-recursive).

    Returns absolute paths.
    """
    if not os.path.isdir(folder):
        return []
    files = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and is_audio_file(path):
            files.append(path)
    return files


def ensure_folder(folder: str):
    """Create folder if it doesn't exist."""
    os.makedirs(folder, exist_ok=True)


def generate_audio_embedding(audio_path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate audio embedding from an audio file using mel spectrograms.
    
    Steps:
    - Load audio waveform using torchaudio.load
    - Resample to match SAMPLE_RATE if necessary
    - Convert to mono by averaging channels if stereo
    - Compute mel spectrogram
    - Normalize using AmplitudeToDB
    - Collapse along time axis to get 1-dimensional embedding
    
    Returns a 1-D numpy array embedding.
    """
    # Load audio waveform
    waveform, orig_sample_rate = torchaudio.load(audio_path)
    
    # Resample if necessary
    if orig_sample_rate != sample_rate:
        waveform = T.Resample(orig_sample_rate, sample_rate)(waveform)
    
    # Convert to mono by averaging channels
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Convert audio waveform to mel spectrogram
    mel_spectrogram = T.MelSpectrogram(sample_rate=sample_rate)(waveform)
    
    # Normalize the mel spectrogram
    mel_spectrogram = T.AmplitudeToDB()(mel_spectrogram)
    
    # Collapse the mel spectrogram to a 1-dimensional embedding
    embedding = mel_spectrogram.mean(dim=-1).squeeze()
    
    return embedding.numpy()


def semantic_search(folder: str, query_audio_path: str, top_k: int = 5, cache_path: str = None) -> List[dict]:
    """Compute semantic search over audio files in `folder`.

    - If `cache_path` is provided and exists, load embeddings from it (`npz` with arrays 'paths' and 'embs').
    - Otherwise compute embeddings for all audio files in `folder` and optionally save to `cache_path`.
    - Compute embedding for `query_audio_path` and return `top_k` matches sorted by cosine similarity.

    Returns a list of dicts: [{'path': <path>, 'score': <similarity>}, ...]
    """
    # Load or compute gallery embeddings
    audio_paths = list_audio_files(folder)
    if len(audio_paths) == 0:
        return []

    gallery_embs = None
    gallery_paths = audio_paths

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
                e = generate_audio_embedding(p)
                emb_list.append(e)
                print(f"Processed: {os.path.basename(p)}")
            except Exception as ex:
                print(f"Error processing {p}: {ex}")
                emb_list.append(np.zeros(N_MELS, dtype=np.float32))

        gallery_embs = np.vstack([e.reshape(1, -1) for e in emb_list])
        if cache_path:
            try:
                np.savez_compressed(cache_path, paths=np.array(gallery_paths), embs=gallery_embs)
                print(f"Saved embeddings cache to {cache_path}")
            except Exception as ex:
                print(f"Could not save cache: {ex}")

    # Query embedding
    q_e = generate_audio_embedding(query_audio_path).reshape(1, -1)

    # Compute cosine similarities
    sims = cosine_similarity(q_e, gallery_embs)[0]
    idx = np.argsort(-sims)[:top_k]

    results = []
    for i in idx:
        results.append({'path': gallery_paths[i], 'score': float(sims[i])})

    return results


if __name__ == '__main__':
    print('audio_search module loaded')
    print('Embedding type:', EMBEDDING_TYPE)
    print('Embedding model:', EMBEDDING_MODEL_NAME)
    print('Audio backend:', AUDIO_BACKEND)
