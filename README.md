# prompt-refine README.md

* This code will nlp-filter texts from a prompt-file and a nominated set of
directories in which to find summarized documents. 

* before running prompt-refine.py install dependencies by running:
```> pip install -r requirements.txt``` 

* It is also necessary insert the following keys into a .env-file
OPENAI_API_KEY=
SERPAPI_API_KEY=
PINECONE_API_KEY=
PINECONE_ENVIRONMENT=


* The prompt and documents
and/or summaries will be placed in a document-term matrix A and vectorized 
by Singular Value Decomposition as follows:
A = U * S * Vt  where U and V are orthogonal matrices and S is a diagonal
matrix of 'singular values', and Vt is the transpose of V.

* The vector rows of U*S give each document of A as a linear combination 
of 'feature'-vectors which are the rows of Vt.

* the semantic metric is cosine-similarity which ignores ther magnitude of the
vectors of U*S 


The prompt-file and directories of documents are both specified on the 
command line ( or have defaults) as well as the number of similar vectors 
used as a context for a new prompt crafted at the end.


* to run:
prompt-refine> python prompt-refine.py [prompt-file-path='./prompt/prompt.txt'] 
   [array of documents directories='corpus/physics.hisy-ph] [context_radius=1]
