# BERT Semantic Search Application

Semantic search engine using BERT embeddings for job descriptions, with multi-modal support for images, audio, and video.

## Features

- **Job Search**: BERT-based semantic search over job descriptions
- **Image Search**: CLIP text-to-image and ResNet-18 visual similarity
- **Audio Search**: Mel spectrogram embeddings
- **Video Search**: CLIP text-to-video search
- **Web Interface**: FastAPI with interactive UI

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git

# Run web server
uvicorn api:app --reload
# Open http://localhost:8000

# Or run CLI job search
python job_search_bert.py job_title_des.csv
```

## Dataset

Download from Kaggle: [Jobs and Job Description Dataset](https://www.kaggle.com/datasets/kshitizregmi/jobs-and-job-description)

```bash
# Using Kaggle API
kaggle datasets download -d kshitizregmi/jobs-and-job-description
unzip jobs-and-job-description.zip
```

## Architecture

```
Query → Text Preprocessing (NLTK) → BERT Embedding (768-dim) → Cosine Similarity → Ranked Results
```

### Models

| Feature | Model | Embedding Dim |
|---------|-------|---------------|
| Job Search | bert-base-uncased | 768 |
| Multi-modal | CLIP ViT-B/32 | 512 |
| Image | ResNet-18 | 512 |
| Audio | Mel Spectrogram | Custom |

## Usage Examples

### Web Interface

Access all search modes at http://localhost:8000 after starting the server.

### CLI Job Search

```bash
python job_search_bert.py jobs.csv
# Enter queries like:
# "machine learning engineer with python experience"
# "data scientist SQL analytics"
```

Results show top 5 matches with similarity scores and job descriptions.

## Project Structure

```
search-app/
├── api.py              # FastAPI server
├── job_search_bert.py  # BERT semantic search
├── clip_multimodal.py  # CLIP text-to-image/video
├── image_search.py     # ResNet-18 embeddings
├── audio_search.py     # Mel spectrogram search
├── requirements.txt    # Dependencies
└── static/
    └── index.html      # Web UI
```

## Requirements

- Python 3.9+
- PyTorch
- Transformers (Hugging Face)
- FastAPI
- NLTK
- OpenAI CLIP

## Technical Details

**BERT Processing:**
- Preprocessing: Tokenization, stopword removal, lemmatization
- Model: `bert-base-uncased` (110M parameters)
- Extracts [CLS] token embedding (768-dim)
- GPU acceleration with CUDA if available

**Search Performance:**
- First run: Downloads model (~400MB) and generates embeddings
- Subsequent queries: Fast (embeddings cached)
- Handles long descriptions (truncates to 512 tokens)

## Troubleshooting

**Out of Memory:**
- Use CPU instead of GPU for large datasets
- Process in smaller batches

**NLTK Resources:**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

**Model Download:**
- Check internet connection
- Manual download from [Hugging Face](https://huggingface.co/bert-base-uncased)

## References

- [BERT Paper](https://arxiv.org/abs/1810.04805) - Devlin et al., 2018
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Kaggle Dataset](https://www.kaggle.com/datasets/kshitizregmi/jobs-and-job-description)

## License

Educational purposes.
