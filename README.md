# g-ai

A semantic search engine powered by BERT with multi-modal extensions for searching jobs, images, audio, and video.

## Features

- **Job Search**: Semantic search over job descriptions using BERT embeddings
- **Image Search**: Find images using natural language queries (CLIP) or visual similarity (ResNet-18)
- **Audio Search**: Search audio files using mel spectrogram embeddings
- **Video Search**: Text-to-video search using CLIP

## Quick Start

```bash
# Clone the repository
git clone https://github.com/maddymz/g-ai.git
cd g-ai

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r search-app/requirements.txt
pip install git+https://github.com/openai/CLIP.git

# Run the web server
cd search-app && uvicorn api:app --reload
```

Open http://localhost:8000 in your browser.

## Project Structure

```
g-ai/
├── search-app/     # Main application (FastAPI + BERT/CLIP search)
│   ├── api.py              # FastAPI server
│   ├── job_search_bert.py  # BERT semantic search
│   ├── clip_multimodal.py  # CLIP text-to-image/video
│   ├── image_search.py     # ResNet-18 image embeddings
│   ├── audio_search.py     # Audio spectrogram search
│   └── static/index.html   # Web UI
├── learning/       # Educational NLP/ML scripts
├── scripts/        # Test data generation utilities
└── fixtures/       # Sample test data (audio, images, videos)
```

## Architecture

```
Query → Text Preprocessing (NLTK) → BERT Embedding (768-dim) → Cosine Similarity → Ranked Results
```

### Models Used

| Feature | Model | Embedding Dim |
|---------|-------|---------------|
| Job Search | bert-base-uncased | 768 |
| Multi-modal | CLIP ViT-B/32 | 512 |
| Image Search | ResNet-18 | 512 |
| Audio Search | Mel Spectrogram | Custom |

## Usage

### Web Interface

```bash
cd search-app && uvicorn api:app --reload
```

### CLI Job Search

```bash
python search-app/job_search_bert.py search-app/job_title_des.csv
```

### Learning Scripts

```bash
# Chatbot example
python learning/chatbot_basic.py

# Self-attention visualization
python learning/self-attention.py
```

## Requirements

- Python 3.9+
- PyTorch
- Transformers (Hugging Face)
- FastAPI
- NLTK
- OpenAI CLIP (for multi-modal search)

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Required for OpenAI-based scripts |

## License

MIT
