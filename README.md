# Coursera UW Machine Learning Clustering & Retrieval

Course can be found [here](https://www.coursera.org/learn/ml-clustering-and-retrieval)

Notebook for quick search can be found [here]()

Videos in [Bilibili](https://www.bilibili.com/video/av15379838/)(to which I post it) 

- Week 1 Intro

- Week 2 Nearest Neighbor Search: Retrieving Documents
  - Implement nearest neighbor search for retrieval tasks
  - Contrast document representations (e.g., raw word counts, tf-idf,…)
    - Emphasize important words using tf-idf
  - Contrast methods for measuring similarity between two documents
    - Euclidean vs. weighted Euclidean
    - Cosine similarity vs. similarity via unnormalized inner product
  - Describe complexity of brute force search
  - Implement KD-trees for nearest neighbor search
  - Implement LSH for approximate nearest neighbor search
  - Compare pros and cons of KD-trees and LSH, and decide which is more appropriate for given dataset
  - [x] [Choosing features and metrics for nearest neighbor search](https://github.com/SSQ/Coursera-UW-Machine-Learning-Clustering-Retrieval/tree/master/Week%201%20PA%201)
  - [x] [Implementing Locality Sensitive Hashing from scratch](https://github.com/SSQ/Coursera-UW-Machine-Learning-Clustering-Retrieval/tree/master/Week%201%20PA%202)

- Week 3 Clustering with k-means
  - Describe potential applications of clustering
  - Describe the input (unlabeled observations) and output (labels) of a clustering algorithm
  - Determine whether a task is supervised or unsupervised
  - Cluster documents using k-means
  - Interpret k-means as a coordinate descent algorithm
  - Define data parallel problems
  - Explain Map and Reduce steps of MapReduce framework
  - Use existing MapReduce implementations to parallelize kmeans, understanding what’s being done under the hood
  - [x] [Clustering text data with k-means](https://github.com/SSQ/Coursera-UW-Machine-Learning-Clustering-Retrieval/tree/master/Week%203%20PA%201)
  
- Week 4 Mixture Models: Model-Based Clustering
  - Interpret a probabilistic model-based approach to clustering using mixture models
  - Describe model parameters
  - Motivate the utility of soft assignments and describe what they represent
  - Discuss issues related to how the number of parameters grow with the number of dimensions
    - Interpret diagonal covariance versions of mixtures of Gaussians
  - Compare and contrast mixtures of Gaussians and k-means
  - Implement an EM algorithm for inferring soft assignments and cluster parameters
    - Determine an initialization strategy
    - Implement a variant that helps avoid overfitting issues
  - [x] [Implementing EM for Gaussian mixtures](https://github.com/SSQ/Coursera-UW-Machine-Learning-Clustering-Retrieval/tree/master/Week%204%20PA%201)
  - [x] [Clustering text data with Gaussian mixtures](https://github.com/SSQ/Coursera-UW-Machine-Learning-Clustering-Retrieval/tree/master/Week%204%20PA%202)
    
- Week 5 Latent Dirichlet Allocation: Mixed Membership Modeling
  - Compare and contrast clustering and mixed membership models
  - Describe a document clustering model for the bagof-words doc representation
  - Interpret the components of the LDA mixed membership model
  - Analyze a learned LDA model
    - Topics in the corpus
    - Topics per document
  - Describe Gibbs sampling steps at a high level
  - Utilize Gibbs sampling output to form predictions or estimate model parameters
  - Implement collapsed Gibbs sampling for LDA
  - [x] [Modeling text topics with Latent Dirichlet Allocation](https://github.com/SSQ/Coursera-UW-Machine-Learning-Clustering-Retrieval/tree/master/Week%205%20PA%201)
  
- Week 6 Hierarchical Clustering & Closing Remarks
  - Bonus content: Hierarchical clustering
    - Divisive clustering
    - Agglomerative clustering
      - The dendrogram for agglomerative clustering
      - Agglomerative clustering details
  - Hidden Markov models (HMMs): Another notion of “clustering”
  - [x] [Modeling text data with a hierarchy of clusters](https://github.com/SSQ/Coursera-UW-Machine-Learning-Clustering-Retrieval/tree/master/Week%206%20PA%201)
