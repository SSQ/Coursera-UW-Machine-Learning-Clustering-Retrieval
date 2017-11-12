# Implementing EM for Gaussian mixtures
# Goal
- implement the EM algorithm for a Gaussian mixture model
- apply your implementation to cluster images
- explore clustering results and interpret the output of the EM algorithm

# Pipeline
- Create some synthetic data.
- Provide a log likelihood function for this model.
- Implement the EM algorithm.
- Visualize the progress of the parameters during the course of running EM.
- Visualize the convergence of the model.

# File Description
- `.zip` file is data file.
  - [image.sf.zip]() (unzip `image.sf`) consists of 1,328 pages and 7 features.
- description files
  - `.ipynb` file is the solution of Week 4 program assignment 1
    - `3_em-for-gmm.ipynb`
  - `.html` file is the html version of `.ipynb` file.
    - `3_em-for-gmm.html`
  - `.py`
    - `3_em-for-gmm.py`
  - file
    - `3_em-for-gmm`
# Snapshot
- **Recommend** open `md` file inside a file
- open `.html` file via brower for quick look.
# Algorithm
- EM algorithm.
# Implement in details
- Implementing the EM algorithm for Gaussian mixture models
  - E-step: assign cluster responsibilities, given current parameters
    - Drawing from a Gaussian distribution
  - M-step: Update parameters, given current cluster responsibilities.  
    - Computing soft counts
    - Updating weights
    - Updating means
    - Updating covariances.
  - The EM algorithm. 
- Testing the implementation on the simulated data
  - Plot progress of parameters
    1. At initialization (using initial_mu, initial_cov, and initial_weights)
    1. After running the algorithm to completion
    1. After just 12 iterations (using parameters estimates returned when setting maxiter=12)
- Fitting a Gaussian mixture model for image data
  - Initialization
  - Evaluating convergence
  - Evaluating uncertainty. 
  - Interpreting each cluster
  
