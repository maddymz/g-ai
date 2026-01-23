# Job Search with BERT Embeddings

A semantic job search application that uses BERT (Bidirectional Encoder Representations from Transformers) to generate embeddings from job descriptions and perform intelligent semantic search.

## Overview

This application allows you to search for jobs using natural language queries. Instead of simple keyword matching, it uses BERT embeddings to understand the semantic meaning of your query and find the most relevant jobs.

## Features

- **BERT-based Embeddings**: Uses the pretrained `bert-base-uncased` model
- **Semantic Search**: Finds jobs based on meaning, not just keywords
- **Text Preprocessing**: Tokenization, stopword removal, and lemmatization
- **Cosine Similarity**: Ranks results by semantic similarity
- **Interactive CLI**: Easy-to-use command-line interface

## Dataset

The application uses the **Jobs and Job Description** dataset from Kaggle:
- **URL**: https://www.kaggle.com/datasets/kshitizregmi/jobs-and-job-description
- **Content**: Job titles and descriptions across various industries and roles

### Downloading the Dataset

1. Visit the Kaggle dataset page: https://www.kaggle.com/datasets/kshitizregmi/jobs-and-job-description
2. Download the CSV file (requires Kaggle account)
3. Save it in your project directory

**OR** use the Kaggle API:
```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset (requires API credentials configured)
kaggle datasets download -d kshitizregmi/jobs-and-job-description
unzip jobs-and-job-description.zip
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or navigate to the project directory**:
   ```bash
   cd /Users/madhukarraj/vscode/g-ai
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # OR on Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - `numpy` - Numerical computing
   - `pandas` - Data manipulation
   - `torch` - PyTorch deep learning framework
   - `transformers` - Hugging Face BERT models
   - `scikit-learn` - Cosine similarity computation
   - `nltk` - Natural language processing

4. **Download NLTK resources** (automatic on first run):
   The script will automatically download required NLTK data:
   - punkt (tokenizer)
   - stopwords
   - wordnet (lemmatizer)

## Usage

### Basic Usage

Run the script with the path to your dataset:

```bash
python job_search_bert.py path/to/jobs_dataset.csv
```

### Example Session

```bash
# Activate virtual environment
source venv/bin/activate

# Run the application
python job_search_bert.py jobs.csv
```

The application will:
1. Load the dataset
2. Preprocess job descriptions
3. Load the BERT model
4. Generate embeddings for all jobs
5. Start an interactive search interface

### Sample Queries

The application supports natural language queries:

```
Enter your search query: machine learning engineer with python experience

Enter your search query: data scientist SQL analytics

Enter your search query: frontend developer react javascript

Enter your search query: project manager agile scrum certification

Enter your search query: backend developer API microservices

Enter your search query: quit
```

### Search Results

Results include:
- **Similarity Score**: Cosine similarity (0-1, higher is better)
- **Job Title**: The position title
- **Job Description**: Brief description (truncated to 200 chars)

Example output:
```
Top 5 matching jobs:

--- Match #1 (Similarity: 0.8742) ---
Title: Senior Machine Learning Engineer
Description: We are seeking an experienced ML engineer with strong Python skills...

--- Match #2 (Similarity: 0.8521) ---
Title: Data Scientist - ML Platform
Description: Build and deploy machine learning models using Python, TensorFlow...
```

## What Jobs Can Be Searched?

The application can search for jobs across various categories present in the dataset, including:

- **Software Engineering**: Backend, Frontend, Full-stack, Mobile
- **Data Science & Analytics**: Data Scientist, Analyst, ML Engineer
- **DevOps & Infrastructure**: Cloud Engineer, SRE, DevOps
- **Product & Project Management**: Product Manager, Scrum Master
- **Design**: UX/UI Designer, Product Designer
- **Security**: Security Engineer, Cybersecurity Analyst
- **And many more...**

The semantic search understands context, so queries like:
- "python developer with cloud experience" → Finds Python + AWS/Azure/GCP jobs
- "junior data analyst" → Prioritizes entry-level analytics roles
- "remote frontend engineer" → Finds remote-friendly frontend positions

## Technical Details

### How It Works

1. **Text Preprocessing**:
   - Tokenization using NLTK
   - Stopword removal (common words like "the", "is")
   - Lemmatization (reducing words to root form)
   - Punctuation removal

2. **BERT Embedding Generation**:
   - Uses `bert-base-uncased` pretrained model
   - Extracts [CLS] token embedding (768-dimensional vector)
   - Truncates long descriptions to 512 tokens
   - Processes on GPU if available

3. **Semantic Search**:
   - Query is preprocessed and embedded using BERT
   - Cosine similarity computed between query and all job embeddings
   - Top K most similar jobs returned

### Model Information

- **Model**: `bert-base-uncased`
- **Vocabulary Size**: 30,522 tokens
- **Hidden Size**: 768 dimensions
- **Max Sequence Length**: 512 tokens
- **Parameters**: ~110M

### Performance Notes

- **First Run**: Model download (~400MB) and embedding generation takes time
- **GPU Acceleration**: Automatically uses CUDA if available
- **Dataset Size**: Larger datasets take longer to embed (progress shown)
- **Query Speed**: Searches are fast after initial embedding generation

## Troubleshooting

### Out of Memory Error

If you encounter OOM errors with large datasets:
- Reduce batch size by processing jobs one at a time (already implemented)
- Use CPU instead of GPU for smaller memory footprint
- Process dataset in chunks

### Token Length Error

The script sets `truncation=True` to handle descriptions longer than 512 tokens. This is normal and expected.

### NLTK Download Errors

If NLTK resources fail to download automatically:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Model Download Issues

If the BERT model fails to download:
- Check internet connection
- Try manual download from Hugging Face: https://huggingface.co/bert-base-uncased
- Place in `~/.cache/huggingface/transformers/`

## Project Structure

```
g-ai/
├── job_search_bert.py      # Main application script
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── venv/                   # Virtual environment (if created)
└── jobs.csv                # Dataset (download separately)
```

## Future Enhancements

Potential improvements:
- Save/load precomputed embeddings for faster restarts
- Add filtering by job category, location, salary
- Web interface using Flask/FastAPI
- Support for other BERT variants (RoBERTa, DistilBERT)
- Fine-tuning on job-specific corpus
- Export results to JSON/CSV

## References

- **BERT Paper**: [Devlin et al., 2018](https://arxiv.org/abs/1810.04805)
- **Transformers Library**: https://huggingface.co/docs/transformers
- **Dataset**: https://www.kaggle.com/datasets/kshitizregmi/jobs-and-job-description

## License

This project is for educational purposes.
