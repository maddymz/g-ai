### Generating Text Data Embeddings with BERT
You are a senior software engineer experienced in working with large language maodels. Follwo following instructions to write an applicatio nusing python to generate sentence/document embedding using BERT and aloow user to do a semantic search.

## Dataset 
We will use a job title and description dataset to generate text embeddings and perform a semantic search to find jobs matching a given query.
Dataset url: https://www.kaggle.com/datasets/kshitizregmi/jobs-and-job-description 

## Application: Find jobs matching a given query
- Step 1: To generate embeddings, we begin by importing the necessary libraries: numpy, pandas, torch, transformers, sklearn.metrics.pairwise, nltk.tokenize, nltk.corpus, and nltk.stem. These libraries are essential for data processing, deep learning, and similarity computation.

- Step 2: Then, we load our dataset from a CSV file containing job titles and descriptions. After loading the dataset, we preprocess the text data by tokenizing, removing stopwords, punctuation, and lemmatizing.

- Step 3: Then, we load a pretrained BERT tokenizer and model.

- Step 4: We write a function generate_embedding to generate BERT embeddings for each word in the dataset after preprocessing. It tokenizes the job description text, feeds it to BERT, and extracts embedding for it using the [CLS] token embedding.
 - Note: While doing tokenization, we need to set the truncation parameter to True as long descriptions will result in generating an error that the token index sequence length is longer than the specified maximum sequence length for this model, which is 512. Running such a sequence through the model will result in indexing errors. Setting this parameter will truncate the sequences larger than the maximum sequence length the model allows.

- Step 5: We define a function, semantic_search, for performing a semantic search using BERT embeddings. It computes cosine similarity between the query embedding and embeddings of job descriptions to find the most similar jobs.



## Important 
- BERT model to use : Pretrained BERT model (bert-base-uncased)
- user should be able to provide an input to fo the semantic search 
- Also specify that what kind of jobs can be searched by this application based on the embeddings 
