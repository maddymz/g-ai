from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sys
from typing import List, Optional
import uvicorn
import os
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch

# Ensure the search-app directory is on sys.path so we can import job_search_bert
base_dir = os.path.dirname(__file__)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from job_search_bert import preprocess_text, generate_embedding, semantic_search, load_and_preprocess_dataset, load_embeddings
# image search utilities
from image_search import load_image_model, semantic_search as image_semantic_search, preprocess_image, generate_image_embedding


app = FastAPI(title="Job Search API")

# Image gallery configuration
GALLERY_FOLDER = os.path.join(os.path.dirname(__file__), 'image_gallery')
GALLERY_CACHE = os.path.join(os.path.dirname(__file__), 'gallery_embeddings.npz')

# Serve the static HTML UI under /static and provide root that returns index.html
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type='text/html')
    raise HTTPException(status_code=404, detail="Index not found")


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    location: Optional[str] = None
    remote: Optional[bool] = None


@app.on_event("startup")
def startup_event():
    # Load dataset and embeddings from search-app folder
    base = os.path.dirname(__file__)
    csv_path = os.path.join(base, 'job_title_des.csv')
    global df, embeddings, tokenizer, model, device
    df = load_and_preprocess_dataset(csv_path)
    embeddings_path = os.path.join(base, 'embeddings.npz')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.to(device)
    model.eval()
    if os.path.exists(embeddings_path):
        embeddings, _, _ = load_embeddings(embeddings_path)
    else:
        # Generate embeddings if not present (small sample only)
        embs = []
        for text in df['processed_text']:
            embs.append(generate_embedding(text, tokenizer, model, device))
        embeddings = np.array(embs)

    # Load image model on startup for image-search endpoint
    global image_model, image_device
    try:
        image_model, image_device = load_image_model()
    except Exception:
        image_model, image_device = None, None
    
    # Ensure gallery folder exists and populate with test images if empty
    os.makedirs(GALLERY_FOLDER, exist_ok=True)
    fixtures_images_dir = os.path.join(os.path.dirname(base_dir), 'fixtures', 'images')
    if os.path.isdir(fixtures_images_dir) and len(os.listdir(GALLERY_FOLDER)) == 0:
        import shutil
        for fname in os.listdir(fixtures_images_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy(os.path.join(fixtures_images_dir, fname), GALLERY_FOLDER)


@app.post('/search')
def search(req: SearchRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="Query is required")
    results = semantic_search(req.query, embeddings, df, tokenizer, model, device, top_k=req.top_k)
    # Apply filters
    if req.location and 'Location' in df.columns:
        results = results[results['Location'].str.contains(req.location, case=False, na=False)]
    if req.remote is not None and 'Remote' in df.columns:
        results = results[results['Remote'].astype(bool) == req.remote]
    return results.head(req.top_k).to_dict(orient='records')


class ImageSearchRequest(BaseModel):
    # Either provide `path` to an image accessible by the server, or upload via multipart file.
    path: Optional[str] = None
    top_k: Optional[int] = 5




@app.post('/image-search')
async def image_search_endpoint(req: ImageSearchRequest = None, file: UploadFile = File(None)):
    if image_model is None:
        raise HTTPException(status_code=503, detail='Image model not available')

    # If a multipart file is uploaded, save it to a temporary location and search against gallery
    if file is not None:
        # ensure temporary uploads folder
        tmp_dir = os.path.join(base_dir, 'tmp_uploads')
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, file.filename)
        with open(tmp_path, 'wb') as f:
            f.write(await file.read())
        
        # Search against gallery folder with caching
        results = image_semantic_search(
            folder=GALLERY_FOLDER,
            query_image_path=tmp_path,
            model=image_model,
            device=image_device,
            top_k=(req.top_k if req else 5),
            cache_path=GALLERY_CACHE
        )
        
        # Clean up uploaded file
        try:
            os.remove(tmp_path)
        except Exception as e:
            print(f"Warning: Could not remove temp file: {e}")
        
        return {'results': results}

    # Fallback: JSON path provided in request body
    if req is None or not req.path or not os.path.exists(req.path):
        raise HTTPException(status_code=400, detail='Provide a valid image `path` on server or upload a file')

    results = image_semantic_search(
        folder=GALLERY_FOLDER,
        query_image_path=req.path,
        model=image_model,
        device=image_device,
        top_k=req.top_k,
        cache_path=GALLERY_CACHE
    )
    return {'results': results}


@app.get('/gallery')
def get_gallery():
    """List all images in the gallery."""
    images = []
    if os.path.isdir(GALLERY_FOLDER):
        for fname in os.listdir(GALLERY_FOLDER):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                images.append({
                    'filename': fname,
                    'path': os.path.join(GALLERY_FOLDER, fname)
                })
    return {'gallery_folder': GALLERY_FOLDER, 'count': len(images), 'images': images}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
