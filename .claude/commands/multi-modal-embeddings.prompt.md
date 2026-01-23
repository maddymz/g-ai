You are a senior software engineer experienced in working with large language models and CNNs, act based on follwoing instructions.

### Generate multimodal embeddings with a pretrained CLIP model (ViT-B/32)
First, we import the necessary libraries, including clip for the CLIP model, torch for tensor operations, PIL for image handling, pandas for CSV file processing, and os for file and directory manipulation.

After importing libraries, we load the CLIP model and its preprocessing function using clip.load("ViT-B/32") and set the model to evaluation mode using model.eval().

We specify the paths to the folder containing images and the CSV file with image descriptions. 
images folder path: /Users/madhukarraj/Downloads/cbe-vector-databases-main/data/images
images csv path: /Users/madhukarraj/Downloads/cbe-vector-databases-main/data/image_descriptions.csv

We read the CSV file into a data frame. Then, we iterate through each row to find and match image files with their descriptions. We store the matched image paths and descriptions in lists.

We define a function to preprocess images and text descriptions according to the format acceptable by CLIP image and text encoders. The maximum context length in CLIP is 77. We truncate text that exceeds this limit to avoid a limit-exceed error by setting the truncate parameter to True. After preprocessing the images and corresponding text descriptions, we stack the results into tensors for batch processing.

Finally, we generate image and description embeddings using image and text encoder functions provided by the CLIP model and normalize the embeddings to unit length, which helps compare them using cosine similarity.

Here, we save the embeddings, along with the image paths and descriptions, to a file using torch.save(). We will use these embeddings in our search functions to find results that match a given text or image query.

After these steps, we need to implement : 

### Semantic search: Search image by text using CLIP
In the code, we use the embedding we generated above to search for images that are semantically similar to the given text query.

### Semantic search: Search text by image using CLIP
In the code, we use the generated embedding for our dataset to search for semantically similar text based on the given image.