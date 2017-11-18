# Modeling text data with a hierarchy of clusters

# Goal
Hierarchical clustering refers to a class of clustering methods that seek to build a hierarchy of clusters, in which some clusters contain others. In this assignment, we will explore a top-down approach, recursively bipartitioning the data using k-means.

# File Description
- `.zip` file is data file.
  - [people_wiki.gl.zip]() (unzip `people_wiki.gl`) consists of 59,071 samples and 3 features.
- description files
  - `.ipynb` file is the solution of Week 6 program assignment 1
    - `6_hierarchical_clustering.ipynb`
  - `.html` file is the html version of `.ipynb` file.
    - `6_hierarchical_clustering.html`
  - `.py`
    - `6_hierarchical_clustering.py`
    
# Snapshot
- **Recommend** open `md` file 
- open `.html` file via brower for quick look.

# Algorithm
- Divisive clustering.

# Implement in details
- Load the Wikipedia dataset
- Bipartition the Wikipedia dataset using k-means
  - Recall our workflow for clustering text data with k-means:
    1. Load the dataframe containing a dataset, such as the Wikipedia text dataset.
    2. Extract the data matrix from the dataframe.
    3. Run k-means on the data matrix with some value of k.
    4. Visualize the clustering results using the centroids, cluster assignments, and the original dataframe. We keep the original dataframe around because the data matrix does not keep auxiliary information (in the case of the text dataset, the title of each article).
  - Let us modify the workflow to perform bipartitioning:
    1. Load the dataframe containing a dataset, such as the Wikipedia text dataset.
    2. Extract the data matrix from the dataframe.
    3. Run k-means on the data matrix with k=2.
    4. Divide the data matrix into two parts using the cluster assignments.
    5. Divide the dataframe into two parts, again using the cluster assignments. This step is necessary to allow for visualization.
    6. Visualize the bipartition of data.
    We'd like to be able to repeat Steps 3-6 multiple times to produce a hierarchy of clusters
- Visualize the bipartition
- Perform recursive bipartitioning
