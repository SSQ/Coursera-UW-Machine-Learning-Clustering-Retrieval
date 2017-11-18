# Modeling text topics with Latent Dirichlet Allocation

# Goal
- **The focus of this assignment is exploration of results, not implementation**
- apply standard preprocessing techniques on Wikipedia text data
- use GraphLab Create to fit a Latent Dirichlet allocation (LDA) model
- explore and interpret the results, including topic keywords and topic assignments for a document

# Pipeline
- get the top words in each topic and use these to identify topic themes
- predict topic distributions for some example documents
- compare the quality of LDA "nearest neighbors" to the NN output from the first assignment
- understand the role of model hyperparameters alpha and gamma

# File Description
- `.zip` file is data file.
  - [people_wiki.gl.zip]() (unzip `people_wiki.gl`) consists of 59,071 samples and 3 features.
- description files
  - `.ipynb` file is the solution of Week 5 program assignment 1
    - `5_lda_blank.ipynb`
  - `.html` file is the html version of `.ipynb` file.
    - `5_lda_blank.html`
  - `.py`
    - `5_lda_blank.py`
  - file
    - `5_lda_blank`
# Snapshot
- **Recommend** open `md` file inside a file
- open `.html` file via brower for quick look.
# Algorithm
- Latent Dirichlet Allocation algorithm.
# Implement in details
- Text Data Preprocessing
- Model fitting and interpretation
- Load a fitted topic model
- Identifying topic themes by top words
  - Measuring the importance of top words
    - the weights of the top 100 words, sorted by the size
    - the total weight of the top 10 words
- Topic distributions for some example documents  
- Comparing LDA to nearest neighbors for document retrieval
- Understanding the role of LDA model hyperparameters
  - Changing the hyperparameter alpha
    - Here we can clearly see the smoothing enforced by the alpha parameter - notice that when alpha is low most of the weight in the topic distribution for this article goes to a single topic, but when alpha is high the weight is much more evenly distributed across the topics
  - Changing the hyperparameter gamma
    - From these two plots we can see that the low gamma model results in higher weight placed on the top words and lower weight placed on the bottom words for each topic, while the high gamma model places relatively less weight on the top words and more weight on the bottom words. Thus increasing gamma results in topics that have a smoother distribution of weight across all the words in the vocabulary
  
