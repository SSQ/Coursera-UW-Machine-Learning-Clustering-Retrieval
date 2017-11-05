
## Fitting a diagonal covariance Gaussian mixture model to text data

In a previous assignment, we explored k-means clustering for a high-dimensional Wikipedia dataset. We can also model this data with a mixture of Gaussians, though with increasing dimension we run into two important issues associated with using a full covariance matrix for each component.
 * Computational cost becomes prohibitive in high dimensions: score calculations have complexity cubic in the number of dimensions M if the Gaussian has a full covariance matrix.
 * A model with many parameters require more data: observe that a full covariance matrix for an M-dimensional Gaussian will have M(M+1)/2 parameters to fit. With the number of parameters growing roughly as the square of the dimension, it may quickly become impossible to find a sufficient amount of data to make good inferences.

Both of these issues are avoided if we require the covariance matrix of each component to be diagonal, as then it has only M parameters to fit and the score computation decomposes into M univariate score calculations. Recall from the lecture that the M-step for the full covariance is:

\begin{align*}
\hat{\Sigma}_k &= \frac{1}{N_k^{soft}} \sum_{i=1}^N r_{ik} (x_i-\hat{\mu}_k)(x_i - \hat{\mu}_k)^T
\end{align*}

Note that this is a square matrix with M rows and M columns, and the above equation implies that the (v, w) element is computed by

\begin{align*}
\hat{\Sigma}_{k, v, w} &= \frac{1}{N_k^{soft}} \sum_{i=1}^N r_{ik} (x_{iv}-\hat{\mu}_{kv})(x_{iw} - \hat{\mu}_{kw})
\end{align*}

When we assume that this is a diagonal matrix, then non-diagonal elements are assumed to be zero and we only need to compute each of the M elements along the diagonal independently using the following equation. 

\begin{align*}
\hat{\sigma}^2_{k, v} &= \hat{\Sigma}_{k, v, v}  \\
&= \frac{1}{N_k^{soft}} \sum_{i=1}^N r_{ik} (x_{iv}-\hat{\mu}_{kv})^2
\end{align*}

In this section, we will use an EM implementation to fit a Gaussian mixture model with **diagonal** covariances to a subset of the Wikipedia dataset. The implementation uses the above equation to compute each variance term. 

We'll begin by importing the dataset and coming up with a useful representation for each article. After running our algorithm on the data, we will explore the output to see whether we can give a meaningful interpretation to the fitted parameters in our model.

**Note to Amazon EC2 users**: To conserve memory, make sure to stop all the other notebooks before running this notebook.

## Import necessary packages

The following code block will check if you have the correct version of GraphLab Create. Any version later than 1.8.5 will do. To upgrade, read [this page](https://turi.com/download/upgrade-graphlab-create.html).


```python
import sframe                                            # see below for install instruction
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags
from scipy.stats import multivariate_normal
from copy import deepcopy
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
```

We also have a Python file containing implementations for several functions that will be used during the course of this assignment.


```python
wiki = sframe.SFrame('people_wiki.gl/').head(5000)
```

    [INFO] sframe.cython.cy_server: SFrame v2.1 started. Logging C:\Users\SSQ\AppData\Local\Temp\sframe_server_1509884838.log.0
    


```python
wiki.head(2)
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">URI</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">text</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Digby_Morrell&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Digby Morrell</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">digby morrell born 10<br>october 1979 is a former ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Alfred_J._Lewy&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alfred J. Lewy</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">alfred j lewy aka sandy<br>lewy graduated from ...</td>
    </tr>
</table>
[2 rows x 3 columns]<br/>
</div>




```python
def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']
    
    return csr_matrix( (data, indices, indptr), shape)

tf_idf = load_sparse_csr('4_tf_idf.npz')  # NOT people_wiki_tf_idf.npz
map_index_to_word = sframe.SFrame('4_map_index_to_word.gl/')  # NOT people_wiki_map_index_to_word.gl
```


```python
map_index_to_word.head(10)
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">feature</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">category</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">index</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">feature</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">conflating</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">feature</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">diamono</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">feature</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">bailouts</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">feature</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">electionruss</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">feature</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">maywoods</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">feature</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">feduring</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">feature</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">spiderbait</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">feature</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">mcin</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">feature</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">sumiswald</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">feature</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">quinta</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
    </tr>
</table>
[10 rows x 3 columns]<br/>
</div>




```python
tf_idf = normalize(tf_idf)
```

## EM in high dimensions

EM for high-dimensional data requires some special treatment:
 * E step and M step must be vectorized as much as possible, as explicit loops are dreadfully slow in Python.
 * All operations must be cast in terms of sparse matrix operations, to take advantage of computational savings enabled by sparsity of data.
 * Initially, some words may be entirely absent from a cluster, causing the M step to produce zero mean and variance for those words.  This means any data point with one of those words will have 0 probability of being assigned to that cluster since the cluster allows for no variability (0 variance) around that count being 0 (0 mean). Since there is a small chance for those words to later appear in the cluster, we instead assign a small positive variance (~1e-10). Doing so also prevents numerical overflow.
 
We provide the complete implementation for you in the file `em_utilities.py`. For those who are interested, you can read through the code to see how the sparse matrix implementation differs from the previous assignment. 

You are expected to answer some quiz questions using the results of clustering.


```python
def diag(array):
    n = len(array)
    return spdiags(array, 0, n, n)

def logpdf_diagonal_gaussian(x, mean, cov):
    '''
    Compute logpdf of a multivariate Gaussian distribution with diagonal covariance at a given point x.
    A multivariate Gaussian distribution with a diagonal covariance is equivalent
    to a collection of independent Gaussian random variables.

    x should be a sparse matrix. The logpdf will be computed for each row of x.
    mean and cov should be given as 1D numpy arrays
    mean[i] : mean of i-th variable
    cov[i] : variance of i-th variable'''

    n = x.shape[0]
    dim = x.shape[1]
    assert(dim == len(mean) and dim == len(cov))

    # multiply each i-th column of x by (1/(2*sigma_i)), where sigma_i is sqrt of variance of i-th variable.
    scaled_x = x.dot( diag(1./(2*np.sqrt(cov))) )
    # multiply each i-th entry of mean by (1/(2*sigma_i))
    scaled_mean = mean/(2*np.sqrt(cov))

    # sum of pairwise squared Eulidean distances gives SUM[(x_i - mean_i)^2/(2*sigma_i^2)]
    return -np.sum(np.log(np.sqrt(2*np.pi*cov))) - pairwise_distances(scaled_x, [scaled_mean], 'euclidean').flatten()**2
```


```python
def log_sum_exp(x, axis):
    '''Compute the log of a sum of exponentials'''
    x_max = np.max(x, axis=axis)
    if axis == 1:
        return x_max + np.log( np.sum(np.exp(x-x_max[:,np.newaxis]), axis=1) )
    else:
        return x_max + np.log( np.sum(np.exp(x-x_max), axis=0) )

def EM_for_high_dimension(data, means, covs, weights, cov_smoothing=1e-5, maxiter=int(1e3), thresh=1e-4, verbose=False):
    # cov_smoothing: specifies the default variance assigned to absent features in a cluster.
    #                If we were to assign zero variances to absent features, we would be overconfient,
    #                as we hastily conclude that those featurese would NEVER appear in the cluster.
    #                We'd like to leave a little bit of possibility for absent features to show up later.
    n = data.shape[0]
    dim = data.shape[1]
    mu = deepcopy(means)
    Sigma = deepcopy(covs)
    K = len(mu)
    weights = np.array(weights)

    ll = None
    ll_trace = []

    for i in range(maxiter):
        # E-step: compute responsibilities
        logresp = np.zeros((n,K))
        for k in xrange(K):
            logresp[:,k] = np.log(weights[k]) + logpdf_diagonal_gaussian(data, mu[k], Sigma[k])
        ll_new = np.sum(log_sum_exp(logresp, axis=1))
        if verbose:
            print(ll_new)
        logresp -= np.vstack(log_sum_exp(logresp, axis=1))
        resp = np.exp(logresp)
        counts = np.sum(resp, axis=0)

        # M-step: update weights, means, covariances
        weights = counts*1. / np.sum(counts)
        for k in range(K):
            mu[k] = (diag(resp[:,k]).dot(data)).sum(axis=0)/counts[k]
            mu[k] = mu[k].A1

            Sigma[k] = diag(resp[:,k]).dot( data.multiply(data)-2*data.dot(diag(mu[k])) ).sum(axis=0) \
                       + (mu[k]**2)*counts[k]
            Sigma[k] = Sigma[k].A1 / counts[k] + cov_smoothing*np.ones(dim)

        # check for convergence in log-likelihood
        ll_trace.append(ll_new)
        if ll is not None and (ll_new-ll) < thresh and ll_new > -np.inf:
            ll = ll_new
            break
        else:
            ll = ll_new

    out = {'weights':weights,'means':mu,'covs':Sigma,'loglik':ll_trace,'resp':resp}

    return out
```

**Initializing mean parameters using k-means**

Recall from the lectures that EM for Gaussian mixtures is very sensitive to the choice of initial means. With a bad initial set of means, EM may produce clusters that span a large area and are mostly overlapping. To eliminate such bad outcomes, we first produce a suitable set of initial means by using the cluster centers from running k-means.  That is, we first run k-means and then take the final set of means from the converged solution as the initial means in our EM algorithm.


```python
from sklearn.cluster import KMeans

np.random.seed(5)
num_clusters = 25

# Use scikit-learn's k-means to simplify workflow
kmeans_model = KMeans(n_clusters=num_clusters, n_init=5, max_iter=400, random_state=1, n_jobs=-1)
kmeans_model.fit(tf_idf)
centroids, cluster_assignment = kmeans_model.cluster_centers_, kmeans_model.labels_

means = [centroid for centroid in centroids]
```

**Initializing cluster weights**

We will initialize each cluster weight to be the proportion of documents assigned to that cluster by k-means above.


```python
cluster_assignment.shape
```




    (5000L,)




```python
num_docs = tf_idf.shape[0]
weights = []
for i in xrange(num_clusters):
    # Compute the number of data points assigned to cluster i:
    num_assigned = np.sum(cluster_assignment[cluster_assignment==i])
    if num_assigned==0:
        num_assigned=1.
    w = float(num_assigned) / num_docs
    weights.append(w)
```


```python
np.sum(cluster_assignment[cluster_assignment==0])
```




    0




```python
means[0].shape
```




    (100282L,)



**Initializing covariances**

To initialize our covariance parameters, we compute $\hat{\sigma}_{k, j}^2 = \sum_{i=1}^{N}(x_{i,j} - \hat{\mu}_{k, j})^2$ for each feature $j$.  For features with really tiny variances, we assign 1e-8 instead to prevent numerical instability. We do this computation in a vectorized fashion in the following code block.


```python
tf_idf[0]
```




    <1x100282 sparse matrix of type '<type 'numpy.float64'>'
    	with 139 stored elements in Compressed Sparse Row format>




```python
covs = []
for i in xrange(num_clusters):
    member_rows = tf_idf[cluster_assignment==i]
    cov = (member_rows.multiply(member_rows) - 2*member_rows.dot(diag(means[i]))).sum(axis=0).A1 / member_rows.shape[0] \
          + means[i]**2
    cov[cov < 1e-8] = 1e-8
    covs.append(cov)
```

**Running EM**

Now that we have initialized all of our parameters, run EM.


```python
weights
```




    [0.0,
     0.0256,
     0.0636,
     0.0396,
     0.04,
     0.106,
     0.4692,
     0.1974,
     1.3408,
     0.153,
     0.196,
     0.5786,
     0.3192,
     0.611,
     0.672,
     1.245,
     0.2688,
     0.4352,
     0.2988,
     0.9652,
     0.364,
     0.7098,
     0.308,
     1.5824,
     0.8304]




```python
out = EM_for_high_dimension(tf_idf, means, covs, weights, cov_smoothing=1e-10)
print out['loglik'] # print history of log-likelihood over time
```

    [3855857146.0866051, 4844053202.46348, 4844053202.46348]
    

## Interpret clustering results

In contrast to k-means, EM is able to explicitly model clusters of varying sizes and proportions. The relative magnitude of variances in the word dimensions tell us much about the nature of the clusters.

Write yourself a cluster visualizer as follows.  Examining each cluster's mean vector, list the 5 words with the largest mean values (5 most common words in the cluster). For each word, also include the associated variance parameter (diagonal element of the covariance matrix). 

A sample output may be:
```
==========================================================
Cluster 0: Largest mean parameters in cluster 

Word        Mean        Variance    
football    1.08e-01    8.64e-03
season      5.80e-02    2.93e-03
club        4.48e-02    1.99e-03
league      3.94e-02    1.08e-03
played      3.83e-02    8.45e-04
...
```


```python
# Fill in the blanks
def visualize_EM_clusters(means, covs, map_index_to_word):
    print('')
    print('==========================================================')

    num_clusters = len(means)
    for c in xrange(num_clusters):
        print('Cluster {0:d}: Largest mean parameters in cluster '.format(c))
        print('\n{0: <12}{1: <12}{2: <12}'.format('Word', 'Mean', 'Variance'))
        
        # The k'th element of sorted_word_ids should be the index of the word 
        # that has the k'th-largest value in the cluster mean. Hint: Use np.argsort().
        sorted_word_ids = np.argsort(-means[c])

        for i in sorted_word_ids[:5]:
            print '{0: <12}{1:<10.2e}{2:10.2e}'.format(map_index_to_word['category'][i], 
                                                       means[c][i],
                                                       covs[c][i])
        print '\n=========================================================='
```


```python
'''By EM'''
visualize_EM_clusters(out['means'], out['covs'], map_index_to_word)
```

    
    ==========================================================
    Cluster 0: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    minister    7.57e-02    7.42e-03
    election    5.89e-02    3.21e-03
    party       5.89e-02    2.61e-03
    liberal     2.93e-02    4.55e-03
    elected     2.91e-02    8.95e-04
    
    ==========================================================
    Cluster 1: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    film        1.76e-01    6.07e-03
    films       5.50e-02    2.97e-03
    festival    4.66e-02    3.60e-03
    feature     3.69e-02    1.81e-03
    directed    3.39e-02    2.22e-03
    
    ==========================================================
    Cluster 2: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    art         1.26e-01    6.83e-03
    museum      5.62e-02    7.27e-03
    gallery     3.65e-02    3.40e-03
    artist      3.61e-02    1.44e-03
    design      3.20e-02    4.59e-03
    
    ==========================================================
    Cluster 3: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    basketball  1.86e-01    7.78e-03
    nba         1.01e-01    1.22e-02
    points      6.25e-02    5.92e-03
    coach       5.57e-02    5.91e-03
    team        4.68e-02    1.30e-03
    
    ==========================================================
    Cluster 4: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    hockey      2.45e-01    1.64e-02
    nhl         1.56e-01    1.64e-02
    ice         6.40e-02    2.97e-03
    season      5.05e-02    2.52e-03
    league      4.31e-02    1.53e-03
    
    ==========================================================
    Cluster 5: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    republican  7.93e-02    5.20e-03
    senate      5.41e-02    6.28e-03
    house       4.64e-02    2.41e-03
    district    4.60e-02    2.37e-03
    democratic  4.46e-02    3.02e-03
    
    ==========================================================
    Cluster 6: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    she         1.60e-01    4.65e-03
    her         1.00e-01    3.14e-03
    miss        2.22e-02    7.76e-03
    women       1.43e-02    1.36e-03
    womens      1.21e-02    1.46e-03
    
    ==========================================================
    Cluster 7: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    championships7.78e-02    5.17e-03
    m           4.70e-02    7.58e-03
    olympics    4.69e-02    2.59e-03
    medal       4.28e-02    2.44e-03
    she         4.18e-02    5.99e-03
    
    ==========================================================
    Cluster 8: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    book        1.45e-02    9.38e-04
    published   1.23e-02    6.16e-04
    that        1.10e-02    1.73e-04
    novel       1.07e-02    1.43e-03
    he          1.04e-02    6.05e-05
    
    ==========================================================
    Cluster 9: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    she         1.37e-01    4.25e-03
    her         8.99e-02    2.74e-03
    actress     7.65e-02    4.29e-03
    film        5.98e-02    3.44e-03
    drama       5.03e-02    6.40e-03
    
    ==========================================================
    Cluster 10: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    soccer      1.15e-01    2.86e-02
    chess       4.52e-02    1.66e-02
    team        4.13e-02    2.15e-03
    coach       3.09e-02    4.45e-03
    league      3.07e-02    2.01e-03
    
    ==========================================================
    Cluster 11: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    president   2.52e-02    1.29e-03
    chairman    2.44e-02    1.97e-03
    committee   2.34e-02    2.38e-03
    served      2.24e-02    6.99e-04
    executive   2.15e-02    1.23e-03
    
    ==========================================================
    Cluster 12: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    music       7.26e-02    3.48e-03
    jazz        6.07e-02    1.14e-02
    hong        3.78e-02    9.92e-03
    kong        3.50e-02    8.64e-03
    chinese     3.12e-02    5.33e-03
    
    ==========================================================
    Cluster 13: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    university  3.47e-02    8.89e-04
    history     3.38e-02    2.81e-03
    philosophy  2.86e-02    5.35e-03
    professor   2.74e-02    1.08e-03
    studies     2.41e-02    1.95e-03
    
    ==========================================================
    Cluster 14: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    theatre     4.93e-02    6.17e-03
    actor       3.56e-02    2.91e-03
    television  3.21e-02    1.67e-03
    film        2.93e-02    1.16e-03
    comedy      2.86e-02    3.91e-03
    
    ==========================================================
    Cluster 15: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    album       6.76e-02    4.78e-03
    band        5.35e-02    4.21e-03
    music       4.18e-02    1.96e-03
    released    3.13e-02    1.11e-03
    song        2.50e-02    1.81e-03
    
    ==========================================================
    Cluster 16: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    tour        1.14e-01    1.92e-02
    pga         1.08e-01    2.65e-02
    racing      8.45e-02    8.26e-03
    championship6.27e-02    4.54e-03
    formula     6.06e-02    1.31e-02
    
    ==========================================================
    Cluster 17: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    news        5.76e-02    8.06e-03
    radio       5.18e-02    4.62e-03
    show        3.75e-02    2.56e-03
    bbc         3.63e-02    7.41e-03
    chef        3.27e-02    1.18e-02
    
    ==========================================================
    Cluster 18: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    football    1.11e-01    5.60e-03
    yards       7.37e-02    1.72e-02
    nfl         6.98e-02    9.15e-03
    coach       6.74e-02    7.85e-03
    quarterback 4.02e-02    7.16e-03
    
    ==========================================================
    Cluster 19: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    league      5.21e-02    3.13e-03
    club        5.04e-02    2.64e-03
    season      4.77e-02    2.30e-03
    rugby       4.35e-02    8.18e-03
    cup         4.22e-02    2.46e-03
    
    ==========================================================
    Cluster 20: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    orchestra   1.31e-01    1.06e-02
    music       1.23e-01    6.15e-03
    symphony    8.70e-02    1.08e-02
    conductor   8.16e-02    1.01e-02
    philharmonic4.96e-02    3.27e-03
    
    ==========================================================
    Cluster 21: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    law         9.52e-02    8.35e-03
    court       6.84e-02    5.24e-03
    judge       4.59e-02    4.44e-03
    attorney    3.74e-02    4.30e-03
    district    3.72e-02    4.20e-03
    
    ==========================================================
    Cluster 22: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    football    1.21e-01    6.14e-03
    afl         9.58e-02    1.31e-02
    australian  7.91e-02    1.58e-03
    club        5.93e-02    1.76e-03
    season      5.58e-02    1.83e-03
    
    ==========================================================
    Cluster 23: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    research    5.70e-02    2.68e-03
    science     3.50e-02    2.95e-03
    university  3.34e-02    7.14e-04
    professor   3.20e-02    1.26e-03
    physics     2.61e-02    5.43e-03
    
    ==========================================================
    Cluster 24: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    baseball    1.16e-01    5.57e-03
    league      1.03e-01    3.63e-03
    major       5.09e-02    1.19e-03
    games       4.66e-02    1.93e-03
    sox         4.55e-02    6.28e-03
    
    ==========================================================
    

**Quiz Question**. Select all the topics that have a cluster in the model created above. [multiple choice]

## Comparing to random initialization

Create variables for randomly initializing the EM algorithm. Complete the following code block.


```python
np.random.seed(5)
num_clusters = len(means)
num_docs, num_words = tf_idf.shape

random_means = []
random_covs = []
random_weights = []

for k in range(num_clusters):
    
    # Create a numpy array of length num_words with random normally distributed values.
    # Use the standard univariate normal distribution (mean 0, variance 1).
    # YOUR CODE HERE
    mean = np.random.normal(0, 1, num_words)
    
    # Create a numpy array of length num_words with random values uniformly distributed between 1 and 5.
    # YOUR CODE HERE
    cov = np.random.uniform(1, 5, num_words)

    # Initially give each cluster equal weight.
    # YOUR CODE HERE
    weight = num_clusters/float(num_docs)
    
    random_means.append(mean)
    random_covs.append(cov)
    random_weights.append(weight)
```

**Quiz Question**: Try fitting EM with the random initial parameters you created above. (Use `cov_smoothing=1e-5`.) Store the result to `out_random_init`. What is the final loglikelihood that the algorithm converges to? 


```python
print len(means)
print len(random_means)
```

    25
    25
    


```python
out_random_init = EM_for_high_dimension(tf_idf, random_means, random_covs, random_weights, cov_smoothing=1e-5)
```


```python
out_random_init['loglik'][-1]
```




    2362875609.1670547



**Quiz Question:** Is the final loglikelihood larger or smaller than the final loglikelihood we obtained above when initializing EM with the results from running k-means?


```python
out['loglik'][-1]
```




    4844053202.46348




```python
print out_random_init['loglik'][-1]>out['loglik'][-1]
```

    False
    

**Quiz Question**: For the above model, `out_random_init`, use the `visualize_EM_clusters` method you created above. Are the clusters more or less interpretable than the ones found after initializing using k-means?


```python
# YOUR CODE HERE. Use visualize_EM_clusters, which will require you to pass in tf_idf and map_index_to_word.
'''By EM'''
visualize_EM_clusters(out_random_init['means'], out_random_init['covs'], map_index_to_word)
```

    
    ==========================================================
    Cluster 0: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    she         4.21e-02    5.79e-03
    her         2.63e-02    1.82e-03
    music       2.03e-02    2.37e-03
    singapore   1.80e-02    5.72e-03
    bbc         1.20e-02    1.82e-03
    
    ==========================================================
    Cluster 1: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    he          1.38e-02    1.11e-04
    she         1.29e-02    1.67e-03
    university  1.06e-02    3.19e-04
    music       1.05e-02    9.94e-04
    league      1.04e-02    9.58e-04
    
    ==========================================================
    Cluster 2: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    she         3.30e-02    3.96e-03
    her         2.43e-02    2.57e-03
    music       1.46e-02    1.42e-03
    he          1.13e-02    1.20e-04
    festival    1.06e-02    2.02e-03
    
    ==========================================================
    Cluster 3: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    she         2.69e-02    3.29e-03
    her         1.76e-02    1.50e-03
    film        1.43e-02    2.01e-03
    series      1.04e-02    5.20e-04
    he          1.00e-02    9.16e-05
    
    ==========================================================
    Cluster 4: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    she         2.59e-02    3.38e-03
    music       1.54e-02    1.69e-03
    her         1.47e-02    1.28e-03
    he          1.24e-02    1.13e-04
    university  1.05e-02    2.92e-04
    
    ==========================================================
    Cluster 5: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    she         2.73e-02    3.73e-03
    her         2.18e-02    2.47e-03
    league      2.15e-02    2.54e-03
    baseball    1.79e-02    2.28e-03
    season      1.59e-02    9.47e-04
    
    ==========================================================
    Cluster 6: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    she         3.16e-02    3.84e-03
    her         2.92e-02    3.00e-03
    art         1.87e-02    4.07e-03
    league      1.09e-02    1.05e-03
    air         1.08e-02    3.75e-03
    
    ==========================================================
    Cluster 7: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    she         2.67e-02    3.61e-03
    her         1.64e-02    1.50e-03
    university  1.27e-02    4.92e-04
    poker       1.14e-02    6.38e-03
    he          1.10e-02    1.04e-04
    
    ==========================================================
    Cluster 8: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    league      2.46e-02    2.44e-03
    she         1.83e-02    2.55e-03
    season      1.82e-02    1.25e-03
    team        1.66e-02    9.01e-04
    hockey      1.65e-02    6.31e-03
    
    ==========================================================
    Cluster 9: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    she         2.35e-02    2.81e-03
    her         1.70e-02    1.81e-03
    he          1.38e-02    1.23e-04
    university  1.28e-02    3.66e-04
    minister    1.00e-02    1.21e-03
    
    ==========================================================
    Cluster 10: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    she         2.82e-02    3.37e-03
    her         2.11e-02    1.96e-03
    chess       1.55e-02    6.60e-03
    district    1.39e-02    1.71e-03
    university  1.33e-02    4.72e-04
    
    ==========================================================
    Cluster 11: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    she         1.96e-02    2.49e-03
    her         1.49e-02    1.77e-03
    album       1.33e-02    1.98e-03
    music       1.30e-02    9.98e-04
    he          1.30e-02    1.11e-04
    
    ==========================================================
    Cluster 12: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    she         3.42e-02    4.59e-03
    her         2.10e-02    1.98e-03
    film        1.59e-02    2.26e-03
    he          1.20e-02    1.15e-04
    party       1.15e-02    1.04e-03
    
    ==========================================================
    Cluster 13: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    she         1.95e-02    3.20e-03
    he          1.27e-02    9.42e-05
    university  1.27e-02    5.70e-04
    her         1.04e-02    9.32e-04
    president   9.84e-03    3.93e-04
    
    ==========================================================
    Cluster 14: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    law         2.41e-02    5.18e-03
    hong        2.03e-02    6.13e-03
    kong        1.94e-02    5.53e-03
    she         1.48e-02    2.05e-03
    band        1.44e-02    1.93e-03
    
    ==========================================================
    Cluster 15: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    league      1.63e-02    1.76e-03
    she         1.49e-02    2.10e-03
    he          1.35e-02    1.08e-04
    music       1.30e-02    1.10e-03
    season      1.18e-02    8.20e-04
    
    ==========================================================
    Cluster 16: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    film        2.09e-02    3.65e-03
    science     1.26e-02    2.19e-03
    research    1.20e-02    1.17e-03
    music       1.19e-02    1.26e-03
    baseball    1.18e-02    2.13e-03
    
    ==========================================================
    Cluster 17: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    she         2.87e-02    3.67e-03
    orchestra   1.83e-02    4.74e-03
    symphony    1.71e-02    4.89e-03
    music       1.63e-02    1.25e-03
    her         1.54e-02    9.90e-04
    
    ==========================================================
    Cluster 18: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    she         2.50e-02    4.52e-03
    music       2.05e-02    2.45e-03
    he          1.41e-02    1.33e-04
    championship1.26e-02    2.12e-03
    film        1.24e-02    1.67e-03
    
    ==========================================================
    Cluster 19: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    she         2.62e-02    4.08e-03
    he          1.28e-02    1.20e-04
    her         1.22e-02    1.08e-03
    served      1.15e-02    4.49e-04
    taylor      1.13e-02    3.93e-03
    
    ==========================================================
    Cluster 20: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    she         2.37e-02    4.14e-03
    health      1.34e-02    3.64e-03
    her         1.16e-02    9.68e-04
    he          1.15e-02    9.46e-05
    australian  1.13e-02    1.90e-03
    
    ==========================================================
    Cluster 21: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    he          1.35e-02    9.19e-05
    university  1.22e-02    3.46e-04
    she         1.19e-02    1.71e-03
    football    9.61e-03    1.29e-03
    his         9.58e-03    7.85e-05
    
    ==========================================================
    Cluster 22: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    church      1.71e-02    3.31e-03
    she         1.51e-02    2.41e-03
    he          1.39e-02    1.05e-04
    her         1.19e-02    1.47e-03
    music       1.10e-02    1.27e-03
    
    ==========================================================
    Cluster 23: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    she         3.39e-02    5.30e-03
    her         2.08e-02    1.92e-03
    album       1.74e-02    3.15e-03
    music       1.52e-02    1.40e-03
    he          1.19e-02    1.22e-04
    
    ==========================================================
    Cluster 24: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    she         1.73e-02    2.63e-03
    played      1.36e-02    6.66e-04
    football    1.31e-02    2.31e-03
    he          1.22e-02    9.16e-05
    season      1.18e-02    7.78e-04
    
    ==========================================================
    

**Note**: Random initialization may sometimes produce a superior fit than k-means initialization. We do not claim that random initialization is always worse. However, this section does illustrate that random initialization often produces much worse clustering than k-means counterpart. This is the reason why we provide the particular random seed (`np.random.seed(5)`).

## Takeaway

In this assignment we were able to apply the EM algorithm to a mixture of Gaussians model of text data. This was made possible by modifying the model to assume a diagonal covariance for each cluster, and by modifying the implementation to use a sparse matrix representation. In the second part you explored the role of k-means initialization on the convergence of the model as well as the interpretability of the clusters.


```python

```
