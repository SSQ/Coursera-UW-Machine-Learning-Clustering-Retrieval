# Clustering text data with Gaussian mixtures
# Goal
In this section, we will use an EM implementation to fit a Gaussian mixture model with **diagonal** covariances to a subset of the Wikipedia dataset.

# File Description
- `.zip` file is data file.
  - [people_wiki.gl.zip]() (unzip `people_wiki.gl`) consists of 59,071 samples and 3 features.
- description files
  - `.ipynb` file is the solution of Week 4 program assignment 2
    - `4_em-with-text-data.ipynb`
  - `.html` file is the html version of `.ipynb` file.
    - `4_em-with-text-data.html`
  - `.py`
    - `4_em-with-text-data.py`
  - file
    - `4_em-with-text-data`
# Snapshot
- **Recommend** open `md` file inside a file
- open `.html` file via brower for quick look.
# Algorithm
- EM algorithm.
# Implement in details
- Load Wikipedia data and extract TF-IDF features
- EM in high dimensions
  - Log probability function for diagonal covariance Gaussian.
  - EM algorithm for sparse data.
    - Initializing mean parameters using k-means.
    - Initializing cluster weights.
    - Initializing covariances
    - Running EM
- Interpret clusters
- Comparing to random initialization
  
