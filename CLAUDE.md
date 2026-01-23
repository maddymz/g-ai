# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI/ML learning and application project focused on semantic search. The core is a **BERT-based job search engine** with multi-modal extensions (image, audio, video search using CLIP, ResNet, and spectrograms).

## Project Structure

```
g-ai/
├── search-app/     # Main application (FastAPI + BERT/CLIP search)
├── learning/       # Educational NLP/ML scripts
├── scripts/        # Test data generation utilities
├── fixtures/       # Test data (audio/, images/, videos/, cache/)
├── .github/prompts/
├── docs/
└── venv/
```

## Common Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r search-app/requirements.txt
pip install git+https://github.com/openai/CLIP.git  # For CLIP multi-modal search

# Run the FastAPI web server
cd search-app && uvicorn api:app --reload

# Run CLI job search directly
python search-app/job_search_bert.py search-app/job_title_des.csv

# Run educational NLP scripts
python learning/chatbot_basic.py
python learning/self-attention.py

# Generate test data
cd scripts && python generate_gallery_images.py
```

## Architecture

### Main Application (`search-app/`)

```
Query → Text Preprocessing (NLTK) → BERT Embedding (768-dim) → Cosine Similarity → Ranked Results
```

- **api.py**: FastAPI server with endpoints for job, image, audio, and video search
- **job_search_bert.py**: Core BERT semantic search engine using `bert-base-uncased`
- **image_search.py**: ResNet-18 based image embeddings
- **audio_search.py**: Mel spectrogram audio embeddings
- **clip_multimodal.py**: OpenAI CLIP (ViT-B/32) for text-to-image/video search
- **static/index.html**: Web UI

### Learning Scripts (`learning/`)

NLP/ML educational examples: bag-of-words, TF-IDF, Word2Vec (CBOW/skip-gram), GloVe, tokenization, self-attention, and OpenAI API integrations.

### Scripts (`scripts/`)

Test data generation utilities for audio, video, and gallery images.

## Key Technical Details

- **Python 3.9+** with virtual environment at `venv/`
- **Models**: bert-base-uncased (text), ResNet-18 (images), CLIP ViT-B/32 (multi-modal)
- **Embedding caches**: `.npz` and `.pt` files avoid recomputation
- **GPU support**: Automatic CUDA detection with CPU fallback
- **Dataset**: Kaggle job descriptions CSV (`job_title_des.csv`)

## Environment Variables

- `OPENAI_API_KEY` or `API_KEY` - Required for OpenAI-based scripts (chatbot, sentiment analysis)
