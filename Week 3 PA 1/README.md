# Clustering text data with k-means
# Goal
- Cluster Wikipedia documents using k-means
- Explore the role of random initialization on the quality of the clustering
- Explore how results differ after changing the number of clusters
- Evaluate clustering, both quantitatively and qualitatively
# File Description
- `.zip` file is data file.
  - [people_wiki.csv.zip]() (unzip `people_wiki.csv`) consists of 59,071 pages and 3 features. URL name text
- `.json` files 
  - `people_wiki_map_index_to_word.json` 
- `.npz` files
  - `people_wiki_tf_idf.npz`
- description files
  - `.ipynb` file is the solution of Week 3 program assignment 1
    - `2_kmeans-with-text-data_blank.ipynb`
  - `.html` file is the html version of `.ipynb` file.
    - `2_kmeans-with-text-data_blank.html`
  - `.py`
    - `2_kmeans-with-text-data_blank.py`
  - file
    - `2_kmeans-with-text-data_blank`
# Snapshot
- **Recommend** open `md` file inside a file
- open `.html` file via brower for quick look.
# Algorithm
- k-means clustering
- k-means++
# Implement in details
- Load data, extract features
- Normalize all vectors
- Implement k-means
- Revising clusters
- Assessing convergence
- Combining into a single function
- Plotting convergence metric
- Beware of local minima
- How to choose K
- Visualize clusters of documents
- 
  
