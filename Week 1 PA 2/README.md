# Implementing Locality Sensitive Hashing from scratch
# Goal
- Implement the LSH algorithm for approximate nearest neighbor search
- Examine the accuracy for different documents by comparing against brute force search, and also contrast runtimes
- Explore the role of the algorithm’s tuning parameters in the accuracy of the method
# File Description
- `.zip` file is data file.
  - [people_wiki.csv.zip]() (unzip `people_wiki.csv`) consists of 59,071 pages and 3 features. URL name text
- `.json` files 
  - `people_wiki_map_index_to_word.json` 
- `.npz` files
  - `people_wiki_tf_idf.npz`
- description files
  - `.ipynb` file is the solution of Week 2 program assignment 2
    - `1_nearest-neighbors-lsh-implementation.ipynb`
  - `.html` file is the html version of `.ipynb` file.
    - `1_nearest-neighbors-lsh-implementation.html`
  - `.py`
    - `1_nearest-neighbors-lsh-implementation.py`
  - file
    - `1_nearest-neighbors-lsh-implementation`
# Snapshot
- **Recommend** open `md` file inside a file
- open `.html` file via brower for quick look.
# Algorithm
Locality Sensitive Hashing (LSH) 
# Implement in details
- Extract the TF-IDF vectors
- Train an LSH model
- Inspect bins
- Query the LSH model
- Experimenting with your LSH implementation
- Effect of nearby bin search
- Quality metrics for neighbors
- Effect of number of random vectors
  
