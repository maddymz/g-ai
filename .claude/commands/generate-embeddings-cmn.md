### Generating image embeddings
You are a senior software engineer experienced in working with large language models and CNNs. Follow the instructions to write an application using python to generate {{embedding_type}} embedding using {{embedding_model}} and allow user to do a semantic search.

## Imeplementation instructions 
 
## Steps 
**Note:** For {{embedding_type}} embeddings, the code utilizes a pretrained {{embedding_model}} model. 

**Step 1:** We begin by importing necessary libraries. 
```
 For example : os is imported for interacting with the file system, torch for deep learning functionalities, torchvision.transforms for image transformations, torchvision.models for pretrained models, PIL for image processing, and cosine_similarity from sklearn.metrics.pairwise for computing cosine similarity between vectors.
```

**Step 2:** We load a pretrained {{embedding_model}} model. 

**Step 3:** We define a function for preprocessing task
```
 For example: for image processing a method like preprocessing_image , which takes the path to an image file as input and performs a series of transformations on the image using transforms.Compose. These transformations include resizing the image to 256 x 256 pixels, center cropping it to 224 x 224 pixels (as required by ResNet), converting it to a PyTorch tensor, and normalizing its pixel values. Then, we add a batch dimension to the tensor. Torch neural networks expect input in the shape of [batch_size, channels, height, width]. Unsqueezing the tensor changes its shape from [C, H, W] to [1, C, H, W], where 1 is the batch size.
```

**Step 4:** We define a generate function to generate embedding. 
```
 For example:  To generate image embeddings , We define a generate_image_embedding function to generate an embedding for an input image by first preprocessing the image using preprocess_image, passing it through the pretrained {{embedding_model}} model, and then flattening the resulting feature map to obtain a 1-dimensional embedding. The embedding is returned as a NumPy array. 
```

**Step 5:** We define the semantic_search function to perform semantic search based on cosine similarity between a query and embeddings in a specified folder 
```
 For example: For image embedding scenarion - We define the semantic_search function to perform semantic search based on cosine similarity between a query image embedding and embeddings of images in a specified folder. It iterates through each image file in the folder, generates its embedding using generate_image_embedding, computes cosine similarity between the query embedding and the file’s embedding, and stores the file name along with the similarity score in a list. After processing all image files, it sorts the list based on similarity scores in descending order and returns the top N similar images along with their similarity scores.
```

## CRITICAL 
- After evry step of coding - please stop and ask my approval 
