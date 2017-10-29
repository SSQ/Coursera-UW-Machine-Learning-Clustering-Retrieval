
# k-means with text data

In this assignment you will
* Cluster Wikipedia documents using k-means
* Explore the role of random initialization on the quality of the clustering
* Explore how results differ after changing the number of clusters
* Evaluate clustering, both quantitatively and qualitatively

When properly executed, clustering uncovers valuable insights from a set of unlabeled documents.

**Note to Amazon EC2 users**: To conserve memory, make sure to stop all the other notebooks before running this notebook.

## Import necessary packages

The following code block will check if you have the correct version of GraphLab Create. Any version later than 1.8.5 will do. To upgrade, read [this page](https://turi.com/download/upgrade-graphlab-create.html).


```python
#import sframe                                                  # see below for install instruction
import matplotlib.pyplot as plt                                # plotting
import numpy as np                                             # dense matrices
import pandas as pd
import time
import json
from scipy.sparse import csr_matrix                            # sparse matrices
from sklearn.preprocessing import normalize                    # normalizing vectors
from sklearn.metrics import pairwise_distances                 # pairwise distances
import sys      
import os
%matplotlib inline
```


```python
#sframe.get_dependencies()
```

## Load data, extract features

To work with text data, we must first convert the documents into numerical features. As in the first assignment, let's extract TF-IDF features for each article.


```python
def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']

    return csr_matrix( (data, indices, indptr), shape)


wiki = pd.read_csv('people_wiki.csv')
tf_idf = load_sparse_csr('people_wiki_tf_idf.npz')

"""
wiki = sframe.SFrame('people_wiki.gl/')
tf_idf = load_sparse_csr('people_wiki_tf_idf.npz')
map_index_to_word = sframe.SFrame('people_wiki_map_index_to_word.gl/')
"""
```




    "\nwiki = sframe.SFrame('people_wiki.gl/')\ntf_idf = load_sparse_csr('people_wiki_tf_idf.npz')\nmap_index_to_word = sframe.SFrame('people_wiki_map_index_to_word.gl/')\n"




```python
with open('people_wiki_map_index_to_word.json') as people_wiki_map_index_to_word:    
    map_index_to_word = json.load(people_wiki_map_index_to_word)
```

The above matrix contains a TF-IDF score for each of the 59071 pages in the data set and each of the 547979 unique words.

## Normalize all vectors

As discussed in the previous assignment, Euclidean distance can be a poor metric of similarity between documents, as it unfairly penalizes long articles. For a reasonable assessment of similarity, we should disregard the length information and use length-agnostic metrics, such as cosine distance.

The k-means algorithm does not directly work with cosine distance, so we take an alternative route to remove length information: we normalize all vectors to be unit length. It turns out that Euclidean distance closely mimics cosine distance when all vectors are unit length. In particular, the squared Euclidean distance between any two vectors of length one is directly proportional to their cosine distance.

We can prove this as follows. Let $\mathbf{x}$ and $\mathbf{y}$ be normalized vectors, i.e. unit vectors, so that $\|\mathbf{x}\|=\|\mathbf{y}\|=1$. Write the squared Euclidean distance as the dot product of $(\mathbf{x} - \mathbf{y})$ to itself:
\begin{align*}
\|\mathbf{x} - \mathbf{y}\|^2 &= (\mathbf{x} - \mathbf{y})^T(\mathbf{x} - \mathbf{y})\\
                              &= (\mathbf{x}^T \mathbf{x}) - 2(\mathbf{x}^T \mathbf{y}) + (\mathbf{y}^T \mathbf{y})\\
                              &= \|\mathbf{x}\|^2 - 2(\mathbf{x}^T \mathbf{y}) + \|\mathbf{y}\|^2\\
                              &= 2 - 2(\mathbf{x}^T \mathbf{y})\\
                              &= 2(1 - (\mathbf{x}^T \mathbf{y}))\\
                              &= 2\left(1 - \frac{\mathbf{x}^T \mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}\right)\\
                              &= 2\left[\text{cosine distance}\right]
\end{align*}

This tells us that two **unit vectors** that are close in Euclidean distance are also close in cosine distance. Thus, the k-means algorithm (which naturally uses Euclidean distances) on normalized vectors will produce the same results as clustering using cosine distance as a distance metric.

We import the [`normalize()` function](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html) from scikit-learn to normalize all vectors to unit length.


```python
from sklearn.preprocessing import normalize
tf_idf = normalize(tf_idf)
```

## Implement k-means

Let us implement the k-means algorithm. First, we choose an initial set of centroids. A common practice is to choose randomly from the data points.

**Note:** We specify a seed here, so that everyone gets the same answer. In practice, we highly recommend to use different seeds every time (for instance, by using the current timestamp).


```python
def get_initial_centroids(data, k, seed=None):
    '''Randomly choose k data points as initial centroids'''
    if seed is not None: # useful for obtaining consistent results
        np.random.seed(seed)
    n = data.shape[0] # number of data points
        
    # Pick K indices from range [0, N).
    rand_indices = np.random.randint(0, n, k)
    
    # Keep centroids as dense format, as many entries will be nonzero due to averaging.
    # As long as at least one document in a cluster contains a word,
    # it will carry a nonzero weight in the TF-IDF vector of the centroid.
    centroids = data[rand_indices,:].toarray()
    
    return centroids
```

After initialization, the k-means algorithm iterates between the following two steps:
1. Assign each data point to the closest centroid.
$$
z_i \gets \mathrm{argmin}_j \|\mu_j - \mathbf{x}_i\|^2
$$
2. Revise centroids as the mean of the assigned data points.
$$
\mu_j \gets \frac{1}{n_j}\sum_{i:z_i=j} \mathbf{x}_i
$$

In pseudocode, we iteratively do the following:
```
cluster_assignment = assign_clusters(data, centroids)
centroids = revise_centroids(data, k, cluster_assignment)
```

### Assigning clusters

How do we implement Step 1 of the main k-means loop above? First import `pairwise_distances` function from scikit-learn, which calculates Euclidean distances between rows of given arrays. See [this documentation](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances.html) for more information.

For the sake of demonstration, let's look at documents 100 through 102 as query documents and compute the distances between each of these documents and every other document in the corpus. In the k-means algorithm, we will have to compute pairwise distances between the set of centroids and the set of documents.


```python
from sklearn.metrics import pairwise_distances

# Get the TF-IDF vectors for documents 100 through 102.
queries = tf_idf[100:102,:]

# Compute pairwise distances from every data point to each query vector.
dist = pairwise_distances(tf_idf, queries, metric='euclidean')

print dist
```

    [[ 1.41000789  1.36894636]
     [ 1.40935215  1.41023886]
     [ 1.39855967  1.40890299]
     ..., 
     [ 1.41108296  1.39123646]
     [ 1.41022804  1.31468652]
     [ 1.39899784  1.41072448]]
    

More formally, `dist[i,j]` is assigned the distance between the `i`th row of `X` (i.e., `X[i,:]`) and the `j`th row of `Y` (i.e., `Y[j,:]`).

**Checkpoint:** For a moment, suppose that we initialize three centroids with the first 3 rows of `tf_idf`. Write code to compute distances from each of the centroids to all data points in `tf_idf`. Then find the distance between row 430 of `tf_idf` and the second centroid and save it to `dist`.


```python
# Students should write code here
queries = tf_idf[:3,:]

# Compute pairwise distances from every data point to each query vector.
dist = pairwise_distances(tf_idf, queries, metric='euclidean')[430][1]

print dist
```

    1.40713106585
    


```python
'''Test cell'''
if np.allclose(dist, pairwise_distances(tf_idf[430,:], tf_idf[1,:])):
    print('Pass')
else:
    print('Check your code again')
```

    Pass
    

**Checkpoint:** Next, given the pairwise distances, we take the minimum of the distances for each data point. Fittingly, NumPy provides an `argmin` function. See [this documentation](http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.argmin.html) for details.

Read the documentation and write code to produce a 1D array whose i-th entry indicates the centroid that is the closest to the i-th data point. Use the list of distances from the previous checkpoint and save them as `distances`. The value 0 indicates closeness to the first centroid, 1 indicates closeness to the second centroid, and so forth. Save this array as `closest_cluster`.

**Hint:** the resulting array should be as long as the number of data points.


```python
# Students should write code here
distances = pairwise_distances(tf_idf, tf_idf[:3,:], metric='euclidean')
closest_cluster = np.argmin(distances, axis=1)
```


```python
'''Test cell'''
reference = [list(row).index(min(row)) for row in distances]
if np.allclose(closest_cluster, reference):
    print('Pass')
else:
    print('Check your code again')
```

    Pass
    

**Checkpoint:** Let's put these steps together.  First, initialize three centroids with the first 3 rows of `tf_idf`. Then, compute distances from each of the centroids to all data points in `tf_idf`. Finally, use these distance calculations to compute cluster assignments and assign them to `cluster_assignment`.


```python
# Students should write code here
queries = tf_idf[:3,:]
distances = pairwise_distances(tf_idf, queries, metric='euclidean')
cluster_assignment = np.argmin(distances, axis=1)
```


```python
if len(cluster_assignment)==59071 and \
   np.array_equal(np.bincount(cluster_assignment), np.array([23061, 10086, 25924])):
    print('Pass') # count number of data points for each cluster
else:
    print('Check your code again.')
```

    Pass
    

Now we are ready to fill in the blanks in this function:


```python
def assign_clusters(data, centroids):
    
    # Compute distances between each data point and the set of centroids:
    # Fill in the blank (RHS only)
    distances_from_centroids = pairwise_distances(data, centroids, metric='euclidean')
        
    # Compute cluster assignments for each data point:
    # Fill in the blank (RHS only)
    cluster_assignment = np.argmin(distances_from_centroids, axis=1)
    
    return cluster_assignment
```

**Checkpoint**. For the last time, let us check if Step 1 was implemented correctly. With rows 0, 2, 4, and 6 of `tf_idf` as an initial set of centroids, we assign cluster labels to rows 0, 10, 20, ..., and 90 of `tf_idf`. The resulting cluster labels should be `[0, 1, 1, 0, 0, 2, 0, 2, 2, 1]`.


```python
if np.allclose(assign_clusters(tf_idf[0:100:10], tf_idf[0:8:2]), np.array([0, 1, 1, 0, 0, 2, 0, 2, 2, 1])):
    print('Pass')
else:
    print('Check your code again.')
```

    Pass
    

### Revising clusters

Let's turn to Step 2, where we compute the new centroids given the cluster assignments. 

SciPy and NumPy arrays allow for filtering via Boolean masks. For instance, we filter all data points that are assigned to cluster 0 by writing
```
data[cluster_assignment==0,:]
```

To develop intuition about filtering, let's look at a toy example consisting of 3 data points and 2 clusters.


```python
data = np.array([[1., 2., 0.],
                 [0., 0., 0.],
                 [2., 2., 0.]])
centroids = np.array([[0.5, 0.5, 0.],
                      [0., -0.5, 0.]])
```

Let's assign these data points to the closest centroid.


```python
cluster_assignment = assign_clusters(data, centroids)
print cluster_assignment
```

    [0 1 0]
    

The expression `cluster_assignment==1` gives a list of Booleans that says whether each data point is assigned to cluster 1 or not:


```python
cluster_assignment==1
```




    array([False,  True, False], dtype=bool)



Likewise for cluster 0:


```python
cluster_assignment==0
```




    array([ True, False,  True], dtype=bool)



In lieu of indices, we can put in the list of Booleans to pick and choose rows. Only the rows that correspond to a `True` entry will be retained.

First, let's look at the data points (i.e., their values) assigned to cluster 1:


```python
data[cluster_assignment==1]
```




    array([[ 0.,  0.,  0.]])



This makes sense since [0 0 0] is closer to [0 -0.5 0] than to [0.5 0.5 0].

Now let's look at the data points assigned to cluster 0:


```python
data[cluster_assignment==0]
```




    array([[ 1.,  2.,  0.],
           [ 2.,  2.,  0.]])



Again, this makes sense since these values are each closer to [0.5 0.5 0] than to [0 -0.5 0].

Given all the data points in a cluster, it only remains to compute the mean. Use [np.mean()](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.mean.html). By default, the function averages all elements in a 2D array. To compute row-wise or column-wise means, add the `axis` argument. See the linked documentation for details. 

Use this function to average the data points in cluster 0:


```python
data[cluster_assignment==0].mean(axis=0)
```




    array([ 1.5,  2. ,  0. ])



We are now ready to complete this function:


```python
def revise_centroids(data, k, cluster_assignment):
    new_centroids = []
    for i in xrange(k):
        # Select all data points that belong to cluster i. Fill in the blank (RHS only)
        member_data_points = data[cluster_assignment==i]
        # Compute the mean of the data points. Fill in the blank (RHS only)
        centroid = member_data_points.mean(axis=0)
        
        # Convert numpy.matrix type to numpy.ndarray type
        centroid = centroid.A1
        new_centroids.append(centroid)
    new_centroids = np.array(new_centroids)
    
    return new_centroids
```

**Checkpoint**. Let's check our Step 2 implementation. Letting rows 0, 10, ..., 90 of `tf_idf` as the data points and the cluster labels `[0, 1, 1, 0, 0, 2, 0, 2, 2, 1]`, we compute the next set of centroids. Each centroid is given by the average of all member data points in corresponding cluster.


```python
result = revise_centroids(tf_idf[0:100:10], 3, np.array([0, 1, 1, 0, 0, 2, 0, 2, 2, 1]))
if np.allclose(result[0], np.mean(tf_idf[[0,30,40,60]].toarray(), axis=0)) and \
   np.allclose(result[1], np.mean(tf_idf[[10,20,90]].toarray(), axis=0))   and \
   np.allclose(result[2], np.mean(tf_idf[[50,70,80]].toarray(), axis=0)):
    print('Pass')
else:
    print('Check your code')
```

    Pass
    

### Assessing convergence

How can we tell if the k-means algorithm is converging? We can look at the cluster assignments and see if they stabilize over time. In fact, we'll be running the algorithm until the cluster assignments stop changing at all. To be extra safe, and to assess the clustering performance, we'll be looking at an additional criteria: the sum of all squared distances between data points and centroids. This is defined as
$$
J(\mathcal{Z},\mu) = \sum_{j=1}^k \sum_{i:z_i = j} \|\mathbf{x}_i - \mu_j\|^2.
$$
The smaller the distances, the more homogeneous the clusters are. In other words, we'd like to have "tight" clusters.


```python
def compute_heterogeneity(data, k, centroids, cluster_assignment):
    
    heterogeneity = 0.0
    for i in xrange(k):
        
        # Select all data points that belong to cluster i. Fill in the blank (RHS only)
        member_data_points = data[cluster_assignment==i, :]
        
        if member_data_points.shape[0] > 0: # check if i-th cluster is non-empty
            # Compute distances from centroid to data points (RHS only)
            distances = pairwise_distances(member_data_points, [centroids[i]], metric='euclidean')
            squared_distances = distances**2
            heterogeneity += np.sum(squared_distances)
        
    return heterogeneity
```

Let's compute the cluster heterogeneity for the 2-cluster example we've been considering based on our current cluster assignments and centroids.


```python
compute_heterogeneity(data, 2, centroids, cluster_assignment)
```




    7.25



### Combining into a single function

Once the two k-means steps have been implemented, as well as our heterogeneity metric we wish to monitor, it is only a matter of putting these functions together to write a k-means algorithm that

* Repeatedly performs Steps 1 and 2
* Tracks convergence metrics
* Stops if either no assignment changed or we reach a certain number of iterations.


```python
# Fill in the blanks
def kmeans(data, k, initial_centroids, maxiter, record_heterogeneity=None, verbose=False):
    '''This function runs k-means on given data and initial set of centroids.
       maxiter: maximum number of iterations to run.
       record_heterogeneity: (optional) a list, to store the history of heterogeneity as function of iterations
                             if None, do not store the history.
       verbose: if True, print how many data points changed their cluster labels in each iteration'''
    centroids = initial_centroids[:]
    prev_cluster_assignment = None
    
    for itr in xrange(maxiter):        
        if verbose:
            print(itr)
        
        # 1. Make cluster assignments using nearest centroids
        # YOUR CODE HERE
        cluster_assignment = assign_clusters(data, centroids)
            
        # 2. Compute a new centroid for each of the k clusters, averaging all data points assigned to that cluster.
        # YOUR CODE HERE
        centroids = revise_centroids(data, k, cluster_assignment)
            
        # Check for convergence: if none of the assignments changed, stop
        if prev_cluster_assignment is not None and \
          (prev_cluster_assignment==cluster_assignment).all():
            break
        
        # Print number of new assignments 
        if prev_cluster_assignment is not None:
            num_changed = np.sum(prev_cluster_assignment!=cluster_assignment)
            if verbose:
                print('    {0:5d} elements changed their cluster assignment.'.format(num_changed))   
        
        # Record heterogeneity convergence metric
        if record_heterogeneity is not None:
            # YOUR CODE HERE
            score = compute_heterogeneity(data, k, centroids, cluster_assignment)
            record_heterogeneity.append(score)
        
        prev_cluster_assignment = cluster_assignment[:]
        
    return centroids, cluster_assignment
```

## Plotting convergence metric

We can use the above function to plot the convergence metric across iterations.


```python
def plot_heterogeneity(heterogeneity, k):
    plt.figure(figsize=(7,4))
    plt.plot(heterogeneity, linewidth=4)
    plt.xlabel('# Iterations')
    plt.ylabel('Heterogeneity')
    plt.title('Heterogeneity of clustering over time, K={0:d}'.format(k))
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
```

Let's consider running k-means with K=3 clusters for a maximum of 400 iterations, recording cluster heterogeneity at every step.  Then, let's plot the heterogeneity over iterations using the plotting function above.


```python
k = 3
heterogeneity = []
initial_centroids = get_initial_centroids(tf_idf, k, seed=0)
centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400,
                                       record_heterogeneity=heterogeneity, verbose=True)
plot_heterogeneity(heterogeneity, k)
```

    0
    1
        19157 elements changed their cluster assignment.
    2
         7739 elements changed their cluster assignment.
    3
         5119 elements changed their cluster assignment.
    4
         3370 elements changed their cluster assignment.
    5
         2811 elements changed their cluster assignment.
    6
         3233 elements changed their cluster assignment.
    7
         3815 elements changed their cluster assignment.
    8
         3172 elements changed their cluster assignment.
    9
         1149 elements changed their cluster assignment.
    10
          498 elements changed their cluster assignment.
    11
          265 elements changed their cluster assignment.
    12
          149 elements changed their cluster assignment.
    13
          100 elements changed their cluster assignment.
    14
           76 elements changed their cluster assignment.
    15
           67 elements changed their cluster assignment.
    16
           51 elements changed their cluster assignment.
    17
           47 elements changed their cluster assignment.
    18
           40 elements changed their cluster assignment.
    19
           34 elements changed their cluster assignment.
    20
           35 elements changed their cluster assignment.
    21
           39 elements changed their cluster assignment.
    22
           24 elements changed their cluster assignment.
    23
           16 elements changed their cluster assignment.
    24
           12 elements changed their cluster assignment.
    25
           14 elements changed their cluster assignment.
    26
           17 elements changed their cluster assignment.
    27
           15 elements changed their cluster assignment.
    28
           14 elements changed their cluster assignment.
    29
           16 elements changed their cluster assignment.
    30
           21 elements changed their cluster assignment.
    31
           22 elements changed their cluster assignment.
    32
           33 elements changed their cluster assignment.
    33
           35 elements changed their cluster assignment.
    34
           39 elements changed their cluster assignment.
    35
           36 elements changed their cluster assignment.
    36
           36 elements changed their cluster assignment.
    37
           25 elements changed their cluster assignment.
    38
           27 elements changed their cluster assignment.
    39
           25 elements changed their cluster assignment.
    40
           28 elements changed their cluster assignment.
    41
           35 elements changed their cluster assignment.
    42
           31 elements changed their cluster assignment.
    43
           25 elements changed their cluster assignment.
    44
           18 elements changed their cluster assignment.
    45
           15 elements changed their cluster assignment.
    46
           10 elements changed their cluster assignment.
    47
            8 elements changed their cluster assignment.
    48
            8 elements changed their cluster assignment.
    49
            8 elements changed their cluster assignment.
    50
            7 elements changed their cluster assignment.
    51
            8 elements changed their cluster assignment.
    52
            3 elements changed their cluster assignment.
    53
            3 elements changed their cluster assignment.
    54
            4 elements changed their cluster assignment.
    55
            2 elements changed their cluster assignment.
    56
            3 elements changed their cluster assignment.
    57
            3 elements changed their cluster assignment.
    58
            1 elements changed their cluster assignment.
    59
            1 elements changed their cluster assignment.
    60
    


![png](output_70_1.png)


**Quiz Question**. (True/False) The clustering objective (heterogeneity) is non-increasing for this example.

True

**Quiz Question**. Let's step back from this particular example. If the clustering objective (heterogeneity) would ever increase when running k-means, that would indicate: (choose one)

1. k-means algorithm got stuck in a bad local minimum
2. There is a bug in the k-means code
3. All data points consist of exact duplicates
4. Nothing is wrong. The objective should generally go down sooner or later.

4

**Quiz Question**. Which of the cluster contains the greatest number of data points in the end? Hint: Use [`np.bincount()`](http://docs.scipy.org/doc/numpy-1.11.0/reference/generated/numpy.bincount.html) to count occurrences of each cluster label.
 1. Cluster #0
 2. Cluster #1
 3. Cluster #2


```python
np.bincount(cluster_assignment)
```




    array([19595, 10427, 29049], dtype=int64)



3

## Beware of local maxima

One weakness of k-means is that it tends to get stuck in a local minimum. To see this, let us run k-means multiple times, with different initial centroids created using different random seeds.

**Note:** Again, in practice, you should set different seeds for every run. We give you a list of seeds for this assignment so that everyone gets the same answer.

This may take several minutes to run.


```python
k = 10
heterogeneity = {}
import time
start = time.time()
for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:
    initial_centroids = get_initial_centroids(tf_idf, k, seed)
    centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400,
                                           record_heterogeneity=None, verbose=False)
    # To save time, compute heterogeneity only once in the end
    heterogeneity[seed] = compute_heterogeneity(tf_idf, k, centroids, cluster_assignment)
    print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity[seed]))
    sys.stdout.flush()
end = time.time()
print(end-start)
```

    seed=000000, heterogeneity=57457.52442
    seed=020000, heterogeneity=57533.20100
    seed=040000, heterogeneity=57512.69257
    seed=060000, heterogeneity=57466.97925
    seed=080000, heterogeneity=57494.92990
    seed=100000, heterogeneity=57484.42210
    seed=120000, heterogeneity=57554.62410
    469.901000023
    

Notice the variation in heterogeneity for different initializations. This indicates that k-means sometimes gets stuck at a bad local minimum.

**Quiz Question**. Another way to capture the effect of changing initialization is to look at the distribution of cluster assignments. Add a line to the code above to compute the size (# of member data points) of clusters for each run of k-means. Look at the size of the largest cluster (most # of member data points) across multiple runs, with seeds 0, 20000, ..., 120000. How much does this measure vary across the runs? What is the minimum and maximum values this quantity takes?


```python
k = 10
heterogeneity = {}
import time
start = time.time()
for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:
    initial_centroids = get_initial_centroids(tf_idf, k, seed)
    centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400,
                                           record_heterogeneity=None, verbose=False)
    
    bin_array = np.bincount(cluster_assignment)
    idx = np.argmax(bin_array) 
    val = bin_array[idx]
    print 'seed={}, max_idx:{}, max_val:{}'.format(seed, idx, val)
    idx = np.argmin(bin_array) 
    val = bin_array[idx]
    print 'seed={}, min_idx:{}, min_val:{}'.format(seed, idx, val)    
    # To save time, compute heterogeneity only once in the end
    heterogeneity[seed] = compute_heterogeneity(tf_idf, k, centroids, cluster_assignment)
    print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity[seed]))
    sys.stdout.flush()
end = time.time()
print(end-start)
```

    seed=0, max_idx:0, max_val:18047
    seed=0, min_idx:4, min_val:1492
    seed=000000, heterogeneity=57457.52442
    seed=20000, max_idx:4, max_val:15779
    seed=20000, min_idx:1, min_val:768
    seed=020000, heterogeneity=57533.20100
    seed=40000, max_idx:9, max_val:18132
    seed=40000, min_idx:2, min_val:186
    seed=040000, heterogeneity=57512.69257
    seed=60000, max_idx:9, max_val:17900
    seed=60000, min_idx:7, min_val:424
    seed=060000, heterogeneity=57466.97925
    seed=80000, max_idx:0, max_val:17582
    seed=80000, min_idx:5, min_val:809
    seed=080000, heterogeneity=57494.92990
    seed=100000, max_idx:4, max_val:16969
    seed=100000, min_idx:1, min_val:1337
    seed=100000, heterogeneity=57484.42210
    seed=120000, max_idx:6, max_val:16481
    seed=120000, min_idx:7, min_val:1608
    seed=120000, heterogeneity=57554.62410
    512.382000208
    

One effective way to counter this tendency is to use **k-means++** to provide a smart initialization. This method tries to spread out the initial set of centroids so that they are not too close together. It is known to improve the quality of local optima and lower average runtime.


```python
def smart_initialize(data, k, seed=None):
    '''Use k-means++ to initialize a good set of centroids'''
    if seed is not None: # useful for obtaining consistent results
        np.random.seed(seed)
    centroids = np.zeros((k, data.shape[1]))
    
    # Randomly choose the first centroid.
    # Since we have no prior knowledge, choose uniformly at random
    idx = np.random.randint(data.shape[0])
    centroids[0] = data[idx,:].toarray()
    # Compute distances from the first centroid chosen to all the other data points
    squared_distances = pairwise_distances(data, centroids[0:1], metric='euclidean').flatten()**2
    
    for i in xrange(1, k):
        # Choose the next centroid randomly, so that the probability for each data point to be chosen
        # is directly proportional to its squared distance from the nearest centroid.
        # Roughtly speaking, a new centroid should be as far as from ohter centroids as possible.
        idx = np.random.choice(data.shape[0], 1, p=squared_distances/sum(squared_distances))
        centroids[i] = data[idx,:].toarray()
        # Now compute distances from the centroids to all data points
        squared_distances = np.min(pairwise_distances(data, centroids[0:i+1], metric='euclidean')**2,axis=1)
    
    return centroids
```

Let's now rerun k-means with 10 clusters using the same set of seeds, but always using k-means++ to initialize the algorithm.

This may take several minutes to run.


```python
k = 10
heterogeneity_smart = {}
start = time.time()
for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:
    initial_centroids = smart_initialize(tf_idf, k, seed)
    centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400,
                                           record_heterogeneity=None, verbose=False)
    # To save time, compute heterogeneity only once in the end
    heterogeneity_smart[seed] = compute_heterogeneity(tf_idf, k, centroids, cluster_assignment)
    print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity_smart[seed]))
    sys.stdout.flush()
end = time.time()
print(end-start)
```

    seed=000000, heterogeneity=57468.63808
    seed=020000, heterogeneity=57486.94263
    seed=040000, heterogeneity=57454.35926
    seed=060000, heterogeneity=57530.43659
    seed=080000, heterogeneity=57454.51852
    seed=100000, heterogeneity=57471.56674
    seed=120000, heterogeneity=57523.28839
    567.506000042
    

Let's compare the set of cluster heterogeneities we got from our 7 restarts of k-means using random initialization compared to the 7 restarts of k-means using k-means++ as a smart initialization.

The following code produces a [box plot](http://matplotlib.org/api/pyplot_api.html) for each of these methods, indicating the spread of values produced by each method.


```python
plt.figure(figsize=(8,5))
plt.boxplot([heterogeneity.values(), heterogeneity_smart.values()], vert=False)
plt.yticks([1, 2], ['k-means', 'k-means++'])
plt.rcParams.update({'font.size': 16})
plt.tight_layout()
```


![png](output_89_0.png)


A few things to notice from the box plot:
* On average, k-means++ produces a better clustering than Random initialization.
* Variation in clustering quality is smaller for k-means++.

**In general, you should run k-means at least a few times with different initializations and then return the run resulting in the lowest heterogeneity.** Let us write a function that runs k-means multiple times and picks the best run that minimizes heterogeneity. The function accepts an optional list of seed values to be used for the multiple runs; if no such list is provided, the current UTC time is used as seed values.


```python
def kmeans_multiple_runs(data, k, maxiter, num_runs, seed_list=None, verbose=False):
    heterogeneity = {}
    
    min_heterogeneity_achieved = float('inf')
    best_seed = None
    final_centroids = None
    final_cluster_assignment = None
    
    for i in xrange(num_runs):
        
        # Use UTC time if no seeds are provided 
        if seed_list is not None: 
            seed = seed_list[i]
            np.random.seed(seed)
        else: 
            seed = int(time.time())
            np.random.seed(seed)
        
        # Use k-means++ initialization
        # YOUR CODE HERE
        initial_centroids = smart_initialize(data, k, seed)
        
        # Run k-means
        # YOUR CODE HERE
        centroids, cluster_assignment = kmeans(data, k, initial_centroids, maxiter, record_heterogeneity=None, verbose=False)
        
        # To save time, compute heterogeneity only once in the end
        # YOUR CODE HERE
        heterogeneity[seed] = compute_heterogeneity(data, k, centroids, cluster_assignment)
        
        if verbose:
            print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity[seed]))
            sys.stdout.flush()
        
        # if current measurement of heterogeneity is lower than previously seen,
        # update the minimum record of heterogeneity.
        if heterogeneity[seed] < min_heterogeneity_achieved:
            min_heterogeneity_achieved = heterogeneity[seed]
            best_seed = seed
            final_centroids = centroids
            final_cluster_assignment = cluster_assignment
    
    # Return the centroids and cluster assignments that minimize heterogeneity.
    return final_centroids, final_cluster_assignment
```

## How to choose K

Since we are measuring the tightness of the clusters, a higher value of K reduces the possible heterogeneity metric by definition.  For example, if we have N data points and set K=N clusters, then we could have 0 cluster heterogeneity by setting the N centroids equal to the values of the N data points. (Note: Not all runs for larger K will result in lower heterogeneity than a single run with smaller K due to local optima.)  Let's explore this general trend for ourselves by performing the following analysis.

Use the `kmeans_multiple_runs` function to run k-means with five different values of K.  For each K, use k-means++ and multiple runs to pick the best solution.  In what follows, we consider K=2,10,25,50,100 and 7 restarts for each setting.

**IMPORTANT: The code block below will take about one hour to finish. We highly suggest that you use the arrays that we have computed for you.**

Side note: In practice, a good implementation of k-means would utilize parallelism to run multiple runs of k-means at once. For an example, see [scikit-learn's KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).


```python
#def plot_k_vs_heterogeneity(k_values, heterogeneity_values):
#    plt.figure(figsize=(7,4))
#    plt.plot(k_values, heterogeneity_values, linewidth=4)
#    plt.xlabel('K')
#    plt.ylabel('Heterogeneity')
#    plt.title('K vs. Heterogeneity')
#    plt.rcParams.update({'font.size': 16})
#    plt.tight_layout()

#start = time.time()
#centroids = {}
#cluster_assignment = {}
#heterogeneity_values = []
#k_list = [2, 10, 25, 50, 100]
#seed_list = [0, 20000, 40000, 60000, 80000, 100000, 120000]

#for k in k_list:
#    heterogeneity = []
#    centroids[k], cluster_assignment[k] = kmeans_multiple_runs(tf_idf, k, maxiter=400,
#                                                               num_runs=len(seed_list),
#                                                               seed_list=seed_list,
#                                                               verbose=True)
#    score = compute_heterogeneity(tf_idf, k, centroids[k], cluster_assignment[k])
#    heterogeneity_values.append(score)

#plot_k_vs_heterogeneity(k_list, heterogeneity_values)

#end = time.time()
#print(end-start)
```

To use the pre-computed NumPy arrays, first download kmeans-arrays.npz as mentioned in the reading for this assignment and load them with the following code.  Make sure the downloaded file is in the same directory as this notebook.


```python
def plot_k_vs_heterogeneity(k_values, heterogeneity_values):
    plt.figure(figsize=(7,4))
    plt.plot(k_values, heterogeneity_values, linewidth=4)
    plt.xlabel('K')
    plt.ylabel('Heterogeneity')
    plt.title('K vs. Heterogeneity')
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

filename = 'kmeans-arrays.npz'

heterogeneity_values = []
k_list = [2, 10, 25, 50, 100]

if os.path.exists(filename):
    arrays = np.load(filename)
    centroids = {}
    cluster_assignment = {}
    for k in k_list:
        print k
        sys.stdout.flush()
        '''To save memory space, do not load the arrays from the file right away. We use
           a technique known as lazy evaluation, where some expressions are not evaluated
           until later. Any expression appearing inside a lambda function doesn't get
           evaluated until the function is called.
           Lazy evaluation is extremely important in memory-constrained setting, such as
           an Amazon EC2 t2.micro instance.'''
        centroids[k] = lambda k=k: arrays['centroids_{0:d}'.format(k)]
        cluster_assignment[k] = lambda k=k: arrays['cluster_assignment_{0:d}'.format(k)]
        score = compute_heterogeneity(tf_idf, k, centroids[k](), cluster_assignment[k]())
        heterogeneity_values.append(score)
    
    plot_k_vs_heterogeneity(k_list, heterogeneity_values)

else:
    print('File not found. Skipping.')
```

    2
    10
    25
    50
    100
    


![png](output_98_1.png)


In the above plot we show that heterogeneity goes down as we increase the number of clusters. Does this mean we should always favor a higher K? **Not at all!** As we will see in the following section, setting K too high may end up separating data points that are actually pretty alike. At the extreme, we can set individual data points to be their own clusters (K=N) and achieve zero heterogeneity, but separating each data point into its own cluster is hardly a desirable outcome. In the following section, we will learn how to detect a K set "too large".

## Visualize clusters of documents

Let's start visualizing some clustering results to see if we think the clustering makes sense.  We can use such visualizations to help us assess whether we have set K too large or too small for a given application.  Following the theme of this course, we will judge whether the clustering makes sense in the context of document analysis.

What are we looking for in a good clustering of documents?
* Documents in the same cluster should be similar.
* Documents from different clusters should be less similar.

So a bad clustering exhibits either of two symptoms:
* Documents in a cluster have mixed content.
* Documents with similar content are divided up and put into different clusters.

To help visualize the clustering, we do the following:
* Fetch nearest neighbors of each centroid from the set of documents assigned to that cluster. We will consider these documents as being representative of the cluster.
* Print titles and first sentences of those nearest neighbors.
* Print top 5 words that have highest tf-idf weights in each centroid.


```python
map_index_to_word = pd.DataFrame(map_index_to_word.items(), columns=['category', 'index'])
```

    
    


```python
wiki.head(2)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>URI</th>
      <th>name</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>&lt;http://dbpedia.org/resource/Digby_Morrell&gt;</td>
      <td>Digby Morrell</td>
      <td>digby morrell born 10 october 1979 is a former...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>&lt;http://dbpedia.org/resource/Alfred_J._Lewy&gt;</td>
      <td>Alfred J. Lewy</td>
      <td>alfred j lewy aka sandy lewy graduated from un...</td>
    </tr>
  </tbody>
</table>
</div>




```python
def visualize_document_clusters(wiki, tf_idf, centroids, cluster_assignment, k, map_index_to_word, display_content=True):
    '''wiki: original dataframe
       tf_idf: data matrix, sparse matrix format
       map_index_to_word: SFrame specifying the mapping betweeen words and column indices
       display_content: if True, display 8 nearest neighbors of each centroid'''
    
    print('==========================================================')

    # Visualize each cluster c
    for c in xrange(k):
        # Cluster heading
        print('Cluster {0:d}    '.format(c)),
        # Print top 5 words with largest TF-IDF weights in the cluster
        idx = centroids[c].argsort()[::-1]
        for i in xrange(5): # Print each word along with the TF-IDF weight
            print('{0:s}:{1:.3f}'.format(map_index_to_word['category'][idx[i]], centroids[c,idx[i]])),
        print('')
        
        if display_content:
            # Compute distances from the centroid to all data points in the cluster,
            # and compute nearest neighbors of the centroids within the cluster.
            distances = pairwise_distances(tf_idf, centroids[c].reshape(1, -1), metric='euclidean').flatten()
            distances[cluster_assignment!=c] = float('inf') # remove non-members from consideration
            nearest_neighbors = distances.argsort()
            # For 8 nearest neighbors, print the title as well as first 180 characters of text.
            # Wrap the text at 80-character mark.
            for i in xrange(8):
                text = ' '.join(wiki.iloc[nearest_neighbors[i]]['text'].split(None, 25)[0:25])
                print('\n* {0:50s} {1:.5f}\n  {2:s}\n  {3:s}'.format(wiki.iloc[nearest_neighbors[i]]['name'],
                    distances[nearest_neighbors[i]], text[:90], text[90:180] if len(text) > 90 else ''))
        print('==========================================================')
```

Let us first look at the 2 cluster case (K=2).


```python
'''Notice the extra pairs of parentheses for centroids and cluster_assignment.
   The centroid and cluster_assignment are still inside the npz file,
   and we need to explicitly indicate when to load them into memory.'''
visualize_document_clusters(wiki, tf_idf, centroids[2](), cluster_assignment[2](), 2, map_index_to_word)
```

    ==========================================================
    Cluster 0     publicationthe:0.025 reapersince:0.017 2009bryant:0.012 gan:0.011 episodesamong:0.011 
    
    * Anita Kunz                                         0.97401
      anita e kunz oc born 1956 is a canadianborn artist and illustratorkunz has lived in london
       new york and toronto contributing to magazines and working
    
    * Janet Jackson                                      0.97472
      janet damita jo jackson born may 16 1966 is an american singer songwriter and actress know
      n for a series of sonically innovative socially conscious and
    
    * Madonna (entertainer)                              0.97475
      madonna louise ciccone tkoni born august 16 1958 is an american singer songwriter actress 
      and businesswoman she achieved popularity by pushing the boundaries of lyrical
    
    * %C3%81ine Hyland                                   0.97536
      ine hyland ne donlon is emeritus professor of education and former vicepresident of univer
      sity college cork ireland she was born in 1942 in athboy co
    
    * Jane Fonda                                         0.97621
      jane fonda born lady jayne seymour fonda december 21 1937 is an american actress writer po
      litical activist former fashion model and fitness guru she is
    
    * Christine Robertson                                0.97643
      christine mary robertson born 5 october 1948 is an australian politician and former austra
      lian labor party member of the new south wales legislative council serving
    
    * Pat Studdy-Clift                                   0.97643
      pat studdyclift is an australian author specialising in historical fiction and nonfictionb
      orn in 1925 she lived in gunnedah until she was sent to a boarding
    
    * Alexandra Potter                                   0.97646
      alexandra potter born 1970 is a british author of romantic comediesborn in bradford yorksh
      ire england and educated at liverpool university gaining an honors degree in
    ==========================================================
    Cluster 1     allmvfc:0.040 genevaswitzerland:0.036 homesbasile:0.029 beakovpov:0.029 hanukkah:0.028 
    
    * Todd Williams                                      0.95468
      todd michael williams born february 13 1971 in syracuse new york is a former major league 
      baseball relief pitcher he attended east syracuseminoa high school
    
    * Gord Sherven                                       0.95622
      gordon r sherven born august 21 1963 in gravelbourg saskatchewan and raised in mankota sas
      katchewan is a retired canadian professional ice hockey forward who played
    
    * Justin Knoedler                                    0.95639
      justin joseph knoedler born july 17 1980 in springfield illinois is a former major league 
      baseball catcherknoedler was originally drafted by the st louis cardinals
    
    * Chris Day                                          0.95648
      christopher nicholas chris day born 28 july 1975 is an english professional footballer who
       plays as a goalkeeper for stevenageday started his career at tottenham
    
    * Tony Smith (footballer, born 1957)                 0.95653
      anthony tony smith born 20 february 1957 is a former footballer who played as a central de
      fender in the football league in the 1970s and
    
    * Ashley Prescott                                    0.95761
      ashley prescott born 11 september 1972 is a former australian rules footballer he played w
      ith the richmond and fremantle football clubs in the afl between
    
    * Leslie Lea                                         0.95802
      leslie lea born 5 october 1942 in manchester is an english former professional footballer 
      he played as a midfielderlea began his professional career with blackpool
    
    * Tommy Anderson (footballer)                        0.95818
      thomas cowan tommy anderson born 24 september 1934 in haddington is a scottish former prof
      essional footballer he played as a forward and was noted for
    ==========================================================
    

Both clusters have mixed content, although cluster 1 is much purer than cluster 0:
* Cluster 0: artists, songwriters, professors, politicians, writers, etc.
* Cluster 1: baseball players, hockey players, soccer (association football) players, etc.

Top words of cluster 1 are all related to sports, whereas top words of cluster 0 show no clear pattern.

Roughly speaking, the entire dataset was divided into athletes and non-athletes. It would be better if we sub-divided non-atheletes into more categories. So let us use more clusters. How about `K=10`?


```python
k = 10
visualize_document_clusters(wiki, tf_idf, centroids[k](), cluster_assignment[k](), k, map_index_to_word)
```

    ==========================================================
    Cluster 0     psihomodo:0.020 himswannack:0.014 gan:0.011 istanbulhatred:0.010 vatslav:0.010 
    
    * Wilson McLean                                      0.97479
      wilson mclean born 1937 is a scottish illustrator and artist he has illustrated primarily 
      in the field of advertising but has also provided cover art
    
    * Anton Hecht                                        0.97748
      anton hecht is an english artist born in london in 2007 he asked musicians from around the
       durham area to contribute to a soundtrack for
    
    * David Salle                                        0.97800
      david salle born 1952 is an american painter printmaker and stage designer who helped defi
      ne postmodern sensibility salle was born in norman oklahoma he earned
    
    * Vipin Sharma                                       0.97805
      vipin sharma is an indian actor born in new delhi he is a graduate of national school of d
      rama new delhi india and the canadian
    
    * Paul Swadel                                        0.97823
      paul swadel is a new zealand film director and producerhe has directed and produced many s
      uccessful short films which have screened in competition at cannes
    
    * Allan Stratton                                     0.97834
      allan stratton born 1951 is a canadian playwright and novelistborn in stratford ontario st
      ratton began his professional arts career while he was still in high
    
    * Bill Bennett (director)                            0.97848
      bill bennett born 1953 is an australian film director producer and screenwriterhe dropped 
      out of medicine at queensland university in 1972 and joined the australian
    
    * Rafal Zielinski                                    0.97850
      rafal zielinski born 1957 montreal is an independent filmmaker he is best known for direct
      ing films such as fun sundance film festival special jury award
    ==========================================================
    Cluster 1     allmvfc:0.052 rcts:0.044 pleckgate:0.042 playerskrush:0.042 genevaswitzerland:0.041 
    
    * Chris Day                                          0.93220
      christopher nicholas chris day born 28 july 1975 is an english professional footballer who
       plays as a goalkeeper for stevenageday started his career at tottenham
    
    * Gary Hooper                                        0.93481
      gary hooper born 26 january 1988 is an english professional footballer who plays as a forw
      ard for norwich cityhooper started his career at nonleague grays
    
    * Tony Smith (footballer, born 1957)                 0.93504
      anthony tony smith born 20 february 1957 is a former footballer who played as a central de
      fender in the football league in the 1970s and
    
    * Jason Roberts (footballer)                         0.93527
      jason andre davis roberts mbe born 25 january 1978 is a former professional footballer and
       now a football punditborn in park royal london roberts was
    
    * Paul Robinson (footballer, born 1979)              0.93587
      paul william robinson born 15 october 1979 is an english professional footballer who plays
       for blackburn rovers as a goalkeeper he is a former england
    
    * Alex Lawless                                       0.93732
      alexander graham alex lawless born 26 march 1985 is a welsh professional footballer who pl
      ays for luton town as a midfielderlawless began his career with
    
    * Neil Grayson                                       0.93748
      neil grayson born 1 november 1964 in york is an english footballer who last played as a st
      riker for sutton towngraysons first club was local
    
    * Sol Campbell                                       0.93759
      sulzeer jeremiah sol campbell born 18 september 1974 is a former england international foo
      tballer a central defender he had a 19year career playing in the
    ==========================================================
    Cluster 2     kluza:0.040 tokyuu:0.037 columnrizzetta:0.032 surmay:0.029 alliancesan:0.029 
    
    * Alessandra Aguilar                                 0.94505
      alessandra aguilar born 1 july 1978 in lugo is a spanish longdistance runner who specialis
      es in marathon running she represented her country in the event
    
    * Heather Samuel                                     0.94529
      heather barbara samuel born 6 july 1970 is a retired sprinter from antigua and barbuda who
       specialized in the 100 and 200 metres in 1990
    
    * Viola Kibiwot                                      0.94617
      viola jelagat kibiwot born december 22 1983 in keiyo district is a runner from kenya who s
      pecialises in the 1500 metres kibiwot won her first
    
    * Ayelech Worku                                      0.94636
      ayelech worku born june 12 1979 is an ethiopian longdistance runner most known for winning
       two world championships bronze medals on the 5000 metres she
    
    * Morhad Amdouni                                     0.94763
      morhad amdouni born 21 january 1988 in portovecchio is a french middle and longdistance ru
      nner he was european junior champion in track and cross country
    
    * Krisztina Papp                                     0.94776
      krisztina papp born 17 december 1982 in eger is a hungarian long distance runner she is th
      e national indoor record holder over 5000 mpapp began
    
    * Petra Lammert                                      0.94869
      petra lammert born 3 march 1984 in freudenstadt badenwrttemberg is a former german shot pu
      tter and current bobsledder she was the 2009 european indoor champion
    
    * Hasan Mahboob                                      0.94880
      hasan mahboob ali born silas kirui on 31 december 1981 in kapsabet is a bahraini longdista
      nce runner he became naturalized in bahrain and switched from
    ==========================================================
    Cluster 3     brilliantly:0.110 allmvfc:0.103 yearsa:0.052 namadi:0.047 genevaswitzerland:0.045 
    
    * Steve Springer                                     0.89300
      steven michael springer born february 11 1961 is an american former professional baseball 
      player who appeared in major league baseball as a third baseman and
    
    * Dave Ford                                          0.89547
      david alan ford born december 29 1956 is a former major league baseball pitcher for the ba
      ltimore orioles born in cleveland ohio ford attended lincolnwest
    
    * Todd Williams                                      0.89820
      todd michael williams born february 13 1971 in syracuse new york is a former major league 
      baseball relief pitcher he attended east syracuseminoa high school
    
    * Justin Knoedler                                    0.90035
      justin joseph knoedler born july 17 1980 in springfield illinois is a former major league 
      baseball catcherknoedler was originally drafted by the st louis cardinals
    
    * Kevin Nicholson (baseball)                         0.90643
      kevin ronald nicholson born march 29 1976 is a canadian baseball shortstop he played part 
      of the 2000 season for the san diego padres of
    
    * James Baldwin (baseball)                           0.90648
      james j baldwin jr born july 15 1971 is a former major league baseball pitcher he batted a
      nd threw righthanded in his 11season career he
    
    * Joe Strong                                         0.90655
      joseph benjamin strong born september 9 1962 in fairfield california is a former major lea
      gue baseball pitcher who played for the florida marlins from 2000
    
    * Javier L%C3%B3pez (baseball)                       0.90691
      javier alfonso lpez born july 11 1977 is a puerto rican professional baseball pitcher for 
      the san francisco giants of major league baseball he is
    ==========================================================
    Cluster 4     flummoxed:0.038 episodesamong:0.035 subpost:0.032 brittbritt:0.023 zunnit:0.019 
    
    * Lawrence W. Green                                  0.95957
      lawrence w green is best known by health education researchers as the originator of the pr
      ecede model and codeveloper of the precedeproceed model which has
    
    * Timothy Luke                                       0.96057
      timothy w luke is university distinguished professor of political science in the college o
      f liberal arts and human sciences as well as program chair of
    
    * Ren%C3%A9e Fox                                     0.96100
      rene c fox a summa cum laude graduate of smith college in 1949 earned her phd in sociology
       in 1954 from radcliffe college harvard university
    
    * Francis Gavin                                      0.96323
      francis j gavin is first frank stanton chair in nuclear security policy studies and profes
      sor of political science at mit before joining mit he was
    
    * Catherine Hakim                                    0.96374
      catherine hakim born 30 may 1948 is a british sociologist who specialises in womens employ
      ment and womens issues she is currently a professorial research fellow
    
    * Stephen Park Turner                                0.96405
      stephen turner is a researcher in social practice social and political theory and the phil
      osophy of the social sciences he is graduate research professor in
    
    * Robert Bates (political scientist)                 0.96489
      robert hinrichs bates born 1942 is an american political scientist he is eaton professor o
      f the science of government in the departments of government and
    
    * Georg von Krogh                                    0.96505
      georg von krogh was born in oslo norway he is a professor at eth zurich and holds the chai
      r of strategic management and innovation he
    ==========================================================
    Cluster 5     beakovpov:0.076 agreements:0.060 cosignatory:0.056 genevaswitzerland:0.044 hanukkah:0.037 
    
    * Todd Curley                                        0.92731
      todd curley born 14 january 1973 is a former australian rules footballer who played for co
      llingwood and the western bulldogs in the australian football league
    
    * Ashley Prescott                                    0.92992
      ashley prescott born 11 september 1972 is a former australian rules footballer he played w
      ith the richmond and fremantle football clubs in the afl between
    
    * Pete Richardson                                    0.93204
      pete richardson born october 17 1946 in youngstown ohio is a former american football defe
      nsive back in the national football league and former college head
    
    * Nathan Brown (Australian footballer born 1976)     0.93561
      nathan daniel brown born 14 august 1976 is an australian rules footballer who played for t
      he melbourne demons in the australian football leaguehe was drafted
    
    * Earl Spalding                                      0.93654
      earl spalding born 11 march 1965 in south perth is a former australian rules footballer wh
      o played for melbourne and carlton in the victorian football
    
    * Bud Grant                                          0.93766
      harry peter bud grant jr born may 20 1927 is a former american football and canadian footb
      all head coach grant served as the head coach
    
    * Tyrone Wheatley                                    0.93885
      tyrone anthony wheatley born january 19 1972 is the running backs coach of michigan and a 
      former professional american football player who played 10 seasons
    
    * Nick Salter                                        0.93916
      nick salter born 30 july 1987 is an australian rules footballer who played for port adelai
      de football club in the australian football league aflhe was
    ==========================================================
    Cluster 6     publicationthe:0.138 reapersince:0.089 tray:0.014 psihomodo:0.013 tourney:0.012 
    
    * Lauren Royal                                       0.93445
      lauren royal born march 3 circa 1965 is a book writer from california royal has written bo
      th historic and novelistic booksa selfproclaimed angels baseball fan
    
    * Barbara Hershey                                    0.93496
      barbara hershey born barbara lynn herzstein february 5 1948 once known as barbara seagull 
      is an american actress in a career spanning nearly 50 years
    
    * Janet Jackson                                      0.93559
      janet damita jo jackson born may 16 1966 is an american singer songwriter and actress know
      n for a series of sonically innovative socially conscious and
    
    * Jane Fonda                                         0.93759
      jane fonda born lady jayne seymour fonda december 21 1937 is an american actress writer po
      litical activist former fashion model and fitness guru she is
    
    * Janine Shepherd                                    0.93833
      janine lee shepherd am born 1962 is an australian pilot and former crosscountry skier shep
      herds career as an athlete ended when she suffered major injuries
    
    * Ellina Graypel                                     0.93847
      ellina graypel born july 19 1972 is an awardwinning russian singersongwriter she was born 
      near the volga river in the heart of russia she spent
    
    * Alexandra Potter                                   0.93858
      alexandra potter born 1970 is a british author of romantic comediesborn in bradford yorksh
      ire england and educated at liverpool university gaining an honors degree in
    
    * Melissa Hart (actress)                             0.93913
      melissa hart is an american actress singer and teacher she made her broadway debut in 1966
       as an ensemble member in jerry bocks the apple
    ==========================================================
    Cluster 7     2009bryant:0.057 longings:0.040 kalyana:0.035 panit:0.023 genets:0.022 
    
    * Brenton Broadstock                                 0.95722
      brenton broadstock ao born 1952 is an australian composerbroadstock was born in melbourne 
      he studied history politics and music at monash university and later composition
    
    * Prince (musician)                                  0.96057
      prince rogers nelson born june 7 1958 known by his mononym prince is an american singerson
      gwriter multiinstrumentalist and actor he has produced ten platinum albums
    
    * Will.i.am                                          0.96066
      william adams born march 15 1975 known by his stage name william pronounced will i am is a
      n american rapper songwriter entrepreneur actor dj record
    
    * Tom Bancroft                                       0.96117
      tom bancroft born 1967 london is a british jazz drummer and composer he began drumming age
      d seven and started off playing jazz with his father
    
    * Julian Knowles                                     0.96152
      julian knowles is an australian composer and performer specialising in new and emerging te
      chnologies his creative work spans the fields of composition for theatre dance
    
    * Dan Siegel (musician)                              0.96223
      dan siegel born in seattle washington is a pianist composer and record producer his earlie
      r music has been described as new age while his more
    
    * Tony Mills (musician)                              0.96238
      tony mills born 7 july 1962 in solihull england is an english rock singer best known for h
      is work with shy and tnthailing from birmingham
    
    * Don Robertson (composer)                           0.96249
      don robertson born 1942 is an american composerdon robertson was born in 1942 in denver co
      lorado and began studying music with conductor and pianist antonia
    ==========================================================
    Cluster 8     airplayramn:0.216 segmentsharma:0.134 embodiments:0.065 genevaswitzerland:0.053 allmvfc:0.047 
    
    * Gord Sherven                                       0.83598
      gordon r sherven born august 21 1963 in gravelbourg saskatchewan and raised in mankota sas
      katchewan is a retired canadian professional ice hockey forward who played
    
    * Eric Brewer                                        0.83765
      eric peter brewer born april 17 1979 is a canadian professional ice hockey defenceman for 
      the anaheim ducks of the national hockey league nhl he
    
    * Stephen Johns (ice hockey)                         0.84580
      stephen johns born april 18 1992 is an american professional ice hockey defenceman he is c
      urrently playing with the rockford icehogs of the american hockey
    
    * Mike Stevens (ice hockey, born 1965)               0.85320
      mike stevens born december 30 1965 in kitchener ontario is a retired professional ice hock
      ey player who played 23 games in the national hockey league
    
    * Tanner Glass                                       0.85484
      tanner glass born november 29 1983 is a canadian professional ice hockey winger who plays 
      for the new york rangers of the national hockey league
    
    * Todd Strueby                                       0.86053
      todd kenneth strueby born june 15 1963 in lanigan saskatchewan and raised in humboldt sask
      atchewan is a retired canadian professional ice hockey centre who played
    
    * Steven King (ice hockey)                           0.86129
      steven andrew king born july 22 1969 in east greenwich rhode island is a former ice hockey
       forward who played professionally from 1991 to 2000
    
    * Don Jackson (ice hockey)                           0.86661
      donald clinton jackson born september 2 1956 in minneapolis minnesota and bloomington minn
      esota is an ice hockey coach and a retired professional ice hockey player
    ==========================================================
    Cluster 9     andmajored:0.028 redningsselskapet:0.025 sslc:0.025 plymouthoriginally:0.021 201021:0.019 
    
    * Doug Lewis                                         0.96516
      douglas grinslade doug lewis pc qc born april 17 1938 is a former canadian politician a ch
      artered accountant and lawyer by training lewis entered the
    
    * David Anderson (British Columbia politician)       0.96530
      david a anderson pc oc born august 16 1937 in victoria british columbia is a former canadi
      an cabinet minister educated at victoria college in victoria
    
    * Lucienne Robillard                                 0.96679
      lucienne robillard pc born june 16 1945 is a canadian politician and a member of the liber
      al party of canada she sat in the house
    
    * Bob Menendez                                       0.96686
      robert bob menendez born january 1 1954 is the senior united states senator from new jerse
      y he is a member of the democratic party first
    
    * Mal Sandon                                         0.96706
      malcolm john mal sandon born 16 september 1945 is an australian politician he was an austr
      alian labor party member of the victorian legislative council from
    
    * Roger Price (Australian politician)                0.96717
      leo roger spurway price born 26 november 1945 is a former australian politician he was ele
      cted as a member of the australian house of representatives
    
    * Maureen Lyster                                     0.96734
      maureen anne lyster born 10 september 1943 is an australian politician she was an australi
      an labor party member of the victorian legislative assembly from 1985
    
    * Don Bell                                           0.96739
      donald h bell born march 10 1942 in new westminster british columbia is a canadian politic
      ian he is currently serving as a councillor for the
    ==========================================================
    

Clusters 0, 1, and 5 appear to be still mixed, but others are quite consistent in content.
* Cluster 0: artists, actors, film directors, playwrights
* Cluster 1: soccer (association football) players, rugby players
* Cluster 2: track and field athletes
* Cluster 3: baseball players
* Cluster 4: professors, researchers, scholars
* Cluster 5: Austrailian rules football players, American football players
* Cluster 6: female figures from various fields
* Cluster 7: composers, songwriters, singers, music producers
* Cluster 8: ice hockey players
* Cluster 9: politicians

Clusters are now more pure, but some are qualitatively "bigger" than others. For instance, the category of scholars is more general than the category of baseball players. Increasing the number of clusters may split larger clusters. Another way to look at the size of the clusters is to count the number of articles in each cluster.


```python
np.bincount(cluster_assignment[10]())
```




    array([17602,  3415,  3535,  1736,  6445,  2552,  7106,  7155,   599,  8926], dtype=int64)



**Quiz Question**. Which of the 10 clusters above contains the greatest number of articles?

1. Cluster 0: artists, actors, film directors, playwrights
2. Cluster 4: professors, researchers, scholars
3. Cluster 5: Austrailian rules football players, American football players
4. Cluster 7: composers, songwriters, singers, music producers
5. Cluster 9: politicians

0

**Quiz Question**. Which of the 10 clusters contains the least number of articles?

1. Cluster 1: soccer (association football) players, rugby players
2. Cluster 3: baseball players
3. Cluster 6: female figures from various fields
4. Cluster 7: composers, songwriters, singers, music producers
5. Cluster 8: ice hockey players

8

There appears to be at least some connection between the topical consistency of a cluster and the number of its member data points.

Let us visualize the case for K=25. For the sake of brevity, we do not print the content of documents. It turns out that the top words with highest TF-IDF weights in each cluster are representative of the cluster.


```python
visualize_document_clusters(wiki, tf_idf, centroids[25](), cluster_assignment[25](), 25,
                            map_index_to_word, display_content=False) # turn off text for brevity
```

    ==========================================================
    Cluster 0     201021:0.077 berlinthat:0.048 discriminao:0.046 cricketmehraj:0.038 qasr:0.038 
    ==========================================================
    Cluster 1     flummoxed:0.054 subpost:0.033 brittbritt:0.032 episodesamong:0.031 thencongresswoman:0.029 
    ==========================================================
    Cluster 2     airplayramn:0.216 segmentsharma:0.134 embodiments:0.065 genevaswitzerland:0.052 allmvfc:0.047 
    ==========================================================
    Cluster 3     andmajored:0.065 redningsselskapet:0.042 5150:0.031 upheavals:0.027 slowburn:0.023 
    ==========================================================
    Cluster 4     respuestas:0.025 gemeenschapscommissiewhen:0.023 cotera:0.022 solleville:0.022 coaccomplice:0.020 
    ==========================================================
    Cluster 5     sslc:0.160 schorrs:0.056 foundationslomax:0.044 andmajored:0.043 redningsselskapet:0.042 
    ==========================================================
    Cluster 6     episodesamong:0.044 subpost:0.037 varaev:0.035 princeweaver:0.034 scriptons:0.031 
    ==========================================================
    Cluster 7     redningsselskapet:0.066 schgerls:0.058 communautaire:0.051 andmajored:0.045 marvelousin:0.043 
    ==========================================================
    Cluster 8     harrisduring:0.095 manitobashe:0.056 columnrizzetta:0.054 creditorsjoo:0.052 tennesseeon:0.051 
    ==========================================================
    Cluster 9     steganography:0.146 polnischen:0.096 imbb:0.053 deens:0.048 flummoxed:0.043 
    ==========================================================
    Cluster 10     kluza:0.075 usaharder:0.050 illinoisstreets:0.048 19401945:0.048 publicationthe:0.048 
    ==========================================================
    Cluster 11     publicationthe:0.144 reapersince:0.092 tourney:0.016 tray:0.015 vatslav:0.012 
    ==========================================================
    Cluster 12     gan:0.011 harcha:0.009 criticsed:0.009 jonsey:0.009 ipfw:0.009 
    ==========================================================
    Cluster 13     brilliantly:0.109 allmvfc:0.104 yearsa:0.052 namadi:0.047 genevaswitzerland:0.045 
    ==========================================================
    Cluster 14     himswannack:0.144 nuig:0.076 mannar:0.056 contenders:0.033 leaded:0.031 
    ==========================================================
    Cluster 15     beakovpov:0.125 bestwritten:0.060 teamsalo:0.051 genevaswitzerland:0.049 hanukkah:0.045 
    ==========================================================
    Cluster 16     2009bryant:0.097 mycologist:0.061 2008kramer:0.033 casesabita:0.029 panit:0.028 
    ==========================================================
    Cluster 17     allmvfc:0.052 rcts:0.044 pleckgate:0.043 playerskrush:0.042 genevaswitzerland:0.042 
    ==========================================================
    Cluster 18     malvinasfalklands:0.055 hebrewaramaicgreek:0.045 istanbulhatred:0.042 oncehallberg:0.039 cotterell:0.035 
    ==========================================================
    Cluster 19     psihomodo:0.095 2003basement:0.038 garcagasco:0.035 allahrakha:0.029 vatslav:0.028 
    ==========================================================
    Cluster 20     longings:0.064 kalyana:0.049 2009bryant:0.037 genets:0.033 violes:0.025 
    ==========================================================
    Cluster 21     mohr:0.075 ltd1999:0.066 tasuku:0.048 elfquestshe:0.047 nedelchev:0.045 
    ==========================================================
    Cluster 22     panit:0.146 fundy:0.116 norvin:0.106 nbthk:0.077 2009bryant:0.064 
    ==========================================================
    Cluster 23     cosignatory:0.120 agreements:0.105 stylecomfollowing:0.065 35acre:0.042 genevaswitzerland:0.040 
    ==========================================================
    Cluster 24     tokyuu:0.256 telenovelami:0.213 fulcrum:0.142 figuresross:0.073 vammala:0.062 
    ==========================================================
    

Looking at the representative examples and top words, we classify each cluster as follows. Notice the bolded items, which indicate the appearance of a new theme.
* Cluster 0: **lawyers, judges, legal scholars**
* Cluster 1: **professors, researchers, scholars (natural and health sciences)**
* Cluster 2: ice hockey players
* Cluster 3: politicans
* Cluster 4: **government officials**
* Cluster 5: politicans
* Cluster 6: **professors, researchers, scholars (social sciences and humanities)**
* Cluster 7: Canadian politicians
* Cluster 8: **car racers**
* Cluster 9: **economists**
* Cluster 10: track and field athletes
* Cluster 11: females from various fields
* Cluster 12: (mixed; no clear theme)
* Cluster 13: baseball players
* Cluster 14: **painters, sculptors, artists**
* Cluster 15: Austrailian rules football players, American football players
* Cluster 16: **musicians, composers**
* Cluster 17: soccer (association football) players, rugby players
* Cluster 18: **poets**
* Cluster 19: **film directors, playwrights**
* Cluster 20: **songwriters, singers, music producers**
* Cluster 21: **generals of U.S. Air Force**
* Cluster 22: **music directors, conductors**
* Cluster 23: **basketball players**
* Cluster 24: **golf players**

Indeed, increasing K achieved the desired effect of breaking up large clusters.  Depending on the application, this may or may not be preferable to the K=10 analysis.

Let's take it to the extreme and set K=100. We have a suspicion that this value is too large. Let us look at the top words from each cluster:


```python
k=100
visualize_document_clusters(wiki, tf_idf, centroids[k](), cluster_assignment[k](), k,
                            map_index_to_word, display_content=False)
# turn off text for brevity -- turn it on if you are curious ;)
```

    ==========================================================
    Cluster 0     shuntaro:0.137 amsterdamhilus:0.082 rezas:0.056 moneyasyougroworg:0.053 coatsworth:0.050 
    ==========================================================
    Cluster 1     mohr:0.170 atthe:0.085 cochampions:0.083 elfquestshe:0.072 legitimatefollowing:0.058 
    ==========================================================
    Cluster 2     wolpemaintained:0.247 examinationspaulker:0.069 koula:0.056 shimomura:0.031 19861992:0.029 
    ==========================================================
    Cluster 3     passive:0.181 likeminded:0.121 1375:0.042 loaizaceballos:0.036 istanbulhatred:0.034 
    ==========================================================
    Cluster 4     gtrafter:0.309 chooky:0.220 kiruminzoo:0.066 mushaima:0.041 tournamentoutscoring:0.031 
    ==========================================================
    Cluster 5     flourish:0.192 harcha:0.127 jjamz:0.054 criticsed:0.046 chaovana:0.042 
    ==========================================================
    Cluster 6     qasr:0.059 berlinthat:0.053 israelthe:0.051 wkiewdekwrza:0.049 dieguito:0.044 
    ==========================================================
    Cluster 7     thenama:0.105 ebo:0.099 communautaire:0.071 redningsselskapet:0.067 rhinehartwringham:0.061 
    ==========================================================
    Cluster 8     steganography:0.065 episodesamong:0.048 flummoxed:0.045 subpost:0.043 polnischen:0.043 
    ==========================================================
    Cluster 9     quvenzhan:0.086 knop:0.076 recordingstheir:0.061 crash:0.053 adhunik:0.040 
    ==========================================================
    Cluster 10     publicationthe:0.188 reapersince:0.052 hawkcurrently:0.026 lectrices:0.020 goodhe:0.019 
    ==========================================================
    Cluster 11     gvanim:0.246 ylf:0.097 william:0.081 usaharder:0.073 kluza:0.068 
    ==========================================================
    Cluster 12     famegrinold:0.086 radical:0.085 amsterdamharry:0.057 bacheva:0.038 himswannack:0.025 
    ==========================================================
    Cluster 13     cricketmehraj:0.098 sharewarejunkiescom:0.051 berlinthat:0.044 redningsselskapet:0.043 qasr:0.043 
    ==========================================================
    Cluster 14     panit:0.227 norvin:0.177 abalos:0.084 2009bryant:0.080 nbthk:0.057 
    ==========================================================
    Cluster 15     ltd1999:0.375 tasuku:0.242 nedelchev:0.106 siveco:0.094 grens:0.080 
    ==========================================================
    Cluster 16     brilliantly:0.098 allmvfc:0.097 312prior:0.083 fifteentime:0.083 sturgesssturgess:0.075 
    ==========================================================
    Cluster 17     elfquestshe:0.114 turova:0.072 indianapolislanzendorf:0.066 worlddr:0.047 skydiving:0.037 
    ==========================================================
    Cluster 18     violes:0.071 winson:0.043 2009bryant:0.041 longings:0.030 acknowledge:0.025 
    ==========================================================
    Cluster 19     cosignatory:0.165 stylecomfollowing:0.113 fortunate:0.067 genevaswitzerland:0.044 brogliebohm:0.044 
    ==========================================================
    Cluster 20     himswannack:0.209 nuig:0.186 mannar:0.082 leaded:0.046 solutionhe:0.044 
    ==========================================================
    Cluster 21     malvinasfalklands:0.213 ageslack:0.083 prou:0.069 94year:0.044 spices:0.040 
    ==========================================================
    Cluster 22     earnestness:0.215 capitalizationmr:0.045 2009bryant:0.045 parlamentarisme:0.037 vietnameseamerican:0.028 
    ==========================================================
    Cluster 23     hebrewaramaicgreek:0.127 oncehallberg:0.045 ondrek:0.044 istanbulhatred:0.039 cotterell:0.030 
    ==========================================================
    Cluster 24     mycologist:0.205 2009bryant:0.048 kalyana:0.034 pflanzgarten:0.025 seventywong:0.023 
    ==========================================================
    Cluster 25     ghenteeklo:0.211 pensioninadvance:0.097 1978041717:0.091 2008durfee:0.039 publicationthe:0.023 
    ==========================================================
    Cluster 26     saidcables:0.259 richardcarl:0.178 sexcolumnist:0.058 klagbsrun:0.033 fearfulon:0.027 
    ==========================================================
    Cluster 27     tokyuu:0.261 telenovelami:0.220 fulcrum:0.140 figuresross:0.073 vammala:0.063 
    ==========================================================
    Cluster 28     bestwritten:0.177 beakovpov:0.128 ebo:0.092 rocksince:0.064 genevaswitzerland:0.062 
    ==========================================================
    Cluster 29     ashot:0.263 wapnick:0.107 actressher:0.095 publicationthe:0.066 embodiments:0.060 
    ==========================================================
    Cluster 30     andmajored:0.073 redningsselskapet:0.035 5150:0.029 kiera:0.022 upheavals:0.021 
    ==========================================================
    Cluster 31     rcts:0.198 playerskrush:0.049 efovi:0.046 hanukkah:0.045 719:0.040 
    ==========================================================
    Cluster 32     istanbulhatred:0.039 incertidumbre:0.029 oncehallberg:0.026 bmias:0.021 minnesotawashington:0.017 
    ==========================================================
    Cluster 33     2008kramer:0.150 2009bryant:0.071 panit:0.056 votebirdinground:0.053 pflanzgarten:0.051 
    ==========================================================
    Cluster 34     pierrepaul:0.299 invertebrate:0.163 uljas:0.092 columnrizzetta:0.079 jocelyne:0.078 
    ==========================================================
    Cluster 35     fundy:0.269 publicationthe:0.067 slovenes:0.041 girliest:0.040 krosnoconstituency:0.036 
    ==========================================================
    Cluster 36     harcha:0.080 criticsed:0.069 burkewhite:0.038 leachon:0.030 vatslav:0.028 
    ==========================================================
    Cluster 37     2009bryant:0.131 stockholmsince:0.038 casesabita:0.037 panit:0.026 schwag:0.023 
    ==========================================================
    Cluster 38     pettibone:0.099 kalyana:0.092 longings:0.040 blasi:0.039 noncommunists:0.034 
    ==========================================================
    Cluster 39     museumhafftka:0.306 protea:0.034 reapersince:0.021 publicationthe:0.020 leachon:0.012 
    ==========================================================
    Cluster 40     cleareyed:0.086 lightbourne:0.072 flummoxed:0.045 brittbritt:0.044 postit:0.042 
    ==========================================================
    Cluster 41     sslc:0.164 schorrs:0.068 foundationslomax:0.043 andmajored:0.039 wahabbism:0.038 
    ==========================================================
    Cluster 42     flummoxed:0.062 subpost:0.035 episodesamong:0.034 brittbritt:0.031 wandgolden:0.030 
    ==========================================================
    Cluster 43     chaovana:0.127 sindre:0.062 5dn:0.059 publicationthe:0.045 harambee:0.045 
    ==========================================================
    Cluster 44     allmvfc:0.088 hechtnielsen:0.060 genevaswitzerland:0.060 pleckgate:0.059 beakovpov:0.055 
    ==========================================================
    Cluster 45     beakovpov:0.046 playerskrush:0.044 pleckgate:0.042 homesbasile:0.041 allmvfc:0.033 
    ==========================================================
    Cluster 46     beakovpov:0.108 availablein:0.099 ebo:0.068 easygoing:0.067 panic:0.064 
    ==========================================================
    Cluster 47     cracra:0.166 schoolnutt:0.119 paternal:0.058 2009tllez:0.038 astralimperial:0.037 
    ==========================================================
    Cluster 48     scriptons:0.227 caretakers:0.045 episodesamong:0.044 subpost:0.041 hilly:0.041 
    ==========================================================
    Cluster 49     thencongresswoman:0.121 foresterhe:0.072 imaginaire:0.060 mustaine:0.053 subpost:0.043 
    ==========================================================
    Cluster 50     niin:0.070 journalismbrooks:0.060 blumstein:0.054 cotera:0.035 5153:0.034 
    ==========================================================
    Cluster 51     martinezlikely:0.143 classifieds:0.136 1999derek:0.095 subbaiah:0.086 sine:0.064 
    ==========================================================
    Cluster 52     cotterell:0.138 happeneda:0.069 videotapein:0.054 cubsdrafted:0.048 kaestneri:0.043 
    ==========================================================
    Cluster 53     tennesseeon:0.477 maymyo:0.121 11155:0.091 lifer:0.078 countriestesak:0.072 
    ==========================================================
    Cluster 54     agito:0.122 earthbased:0.068 californias:0.053 schpfung:0.049 llan:0.028 
    ==========================================================
    Cluster 55     cashforonline:0.282 cepra:0.183 possibleat:0.094 senaratnes:0.046 guernsey:0.027 
    ==========================================================
    Cluster 56     nbthk:0.207 panit:0.136 hinfinity:0.087 2009bryant:0.080 norvin:0.073 
    ==========================================================
    Cluster 57     rotax:0.035 pressurein:0.027 schwan:0.026 discriminao:0.025 mousaviaccording:0.023 
    ==========================================================
    Cluster 58     giannoulias:0.234 kalyana:0.047 2009bryant:0.039 longings:0.037 earnestness:0.035 
    ==========================================================
    Cluster 59     friderick:0.093 wrightson:0.052 beginningsa:0.051 2009bryant:0.048 longings:0.037 
    ==========================================================
    Cluster 60     rezas:0.127 slovenes:0.059 thirdmost:0.035 peersoconnor:0.026 interesses:0.025 
    ==========================================================
    Cluster 61     nagara:0.193 multipledeals:0.132 electrnica:0.052 labelart:0.038 etonian:0.032 
    ==========================================================
    Cluster 62     thisthe:0.362 2011loudon:0.109 1987ali:0.084 publicationthe:0.057 poetryjanice:0.044 
    ==========================================================
    Cluster 63     airplayramn:0.220 segmentsharma:0.138 embodiments:0.067 genevaswitzerland:0.053 allmvfc:0.048 
    ==========================================================
    Cluster 64     201021:0.148 discriminao:0.093 yeniden:0.071 berlinthat:0.051 tram:0.043 
    ==========================================================
    Cluster 65     agreements:0.205 35acre:0.086 cosignatory:0.059 samelands:0.052 beakovpov:0.046 
    ==========================================================
    Cluster 66     undps:0.278 ard:0.168 huhtamo:0.100 vayathinile:0.055 hovrtten:0.031 
    ==========================================================
    Cluster 67     longings:0.088 genets:0.044 2009bryant:0.040 sammelana:0.033 denk:0.027 
    ==========================================================
    Cluster 68     publicationthe:0.158 reapersince:0.152 2009bryant:0.020 longings:0.016 acknowledge:0.013 
    ==========================================================
    Cluster 69     2003basement:0.194 allahrakha:0.034 pollicino:0.031 stihltv:0.029 infobitt:0.027 
    ==========================================================
    Cluster 70     meadowhall:0.099 inequalitiesand:0.089 arema:0.086 flummoxed:0.039 hinaults:0.039 
    ==========================================================
    Cluster 71     pancreasvajo:0.145 upheavals:0.115 andmajored:0.053 slowburn:0.049 wkiewdekwrza:0.048 
    ==========================================================
    Cluster 72     illinoisstreets:0.459 ltd1995:0.087 publicationthe:0.082 reportshe:0.063 kluza:0.062 
    ==========================================================
    Cluster 73     publicationthe:0.147 reapersince:0.105 tray:0.098 psihomodo:0.063 holley:0.054 
    ==========================================================
    Cluster 74     publicationthe:0.101 reapersince:0.065 hawkcurrently:0.012 criticsed:0.010 vatslav:0.009 
    ==========================================================
    Cluster 75     cnie:0.196 elfquestshe:0.177 positionmitchell:0.099 pearlhe:0.074 2009ethan:0.073 
    ==========================================================
    Cluster 76     wappo:0.242 loficahoone:0.064 gifmis:0.061 sslc:0.059 dancejazzdans:0.051 
    ==========================================================
    Cluster 77     psihomodo:0.233 autostraddle:0.085 garcagasco:0.048 severing:0.048 hertzler:0.045 
    ==========================================================
    Cluster 78     tonighttiffin:0.288 dethroned:0.268 neithard:0.068 normal:0.037 emmetts:0.035 
    ==========================================================
    Cluster 79     eightyfive:0.296 allmvfc:0.072 tiotangco:0.065 homesbasile:0.053 genevaswitzerland:0.052 
    ==========================================================
    Cluster 80     gan:0.011 jonsey:0.009 ipfw:0.009 surmay:0.008 hickling:0.007 
    ==========================================================
    Cluster 81     chaisik:0.092 mcculloch:0.072 redningsselskapet:0.072 kannadahe:0.066 twinmaker:0.054 
    ==========================================================
    Cluster 82     kyed:0.048 wwwsusankleinbergcom:0.047 infobitt:0.043 vatslav:0.038 holley:0.037 
    ==========================================================
    Cluster 83     harrisduring:0.128 manitobashe:0.080 creditorsjoo:0.066 disasterin:0.061 institutionon:0.055 
    ==========================================================
    Cluster 84     redningsselskapet:0.096 schgerls:0.086 communautaire:0.071 andmajored:0.067 ismaylov:0.060 
    ==========================================================
    Cluster 85     solleville:0.038 wone:0.031 cotera:0.027 hometownon:0.025 ebn:0.023 
    ==========================================================
    Cluster 86     wellcome:0.414 tsimshianicspeaking:0.085 yolspor:0.066 columnrizzetta:0.064 abaya:0.059 
    ==========================================================
    Cluster 87     mechanicsin:0.077 wwfworld:0.068 908:0.057 contactless:0.048 figuresross:0.047 
    ==========================================================
    Cluster 88     gemeenschapscommissiewhen:0.038 plymouthoriginally:0.028 respuestas:0.028 episodesamong:0.026 branwyn:0.022 
    ==========================================================
    Cluster 89     dens:0.061 rockabillywilliams:0.054 vlei:0.047 cricketmehraj:0.037 1972when:0.037 
    ==========================================================
    Cluster 90     beakovpov:0.120 teamsalo:0.106 substitutetorquay:0.081 whitmarshs:0.052 abcaccording:0.041 
    ==========================================================
    Cluster 91     brilliantly:0.117 allmvfc:0.108 internationalby:0.061 yearsa:0.052 germanytornado:0.044 
    ==========================================================
    Cluster 92     longings:0.115 reapersince:0.073 oncophage:0.066 shoulderfired:0.064 nerys:0.064 
    ==========================================================
    Cluster 93     psihomodo:0.087 garcagasco:0.050 allahrakha:0.029 vatslav:0.024 infobitt:0.022 
    ==========================================================
    Cluster 94     kluza:0.106 19401945:0.086 publicationthe:0.059 alfonsn:0.059 mafyan:0.054 
    ==========================================================
    Cluster 95     himswannack:0.109 mannar:0.040 contenders:0.036 sejmgronkiewiczwaltz:0.032 rigdon:0.032 
    ==========================================================
    Cluster 96     kalyana:0.120 longings:0.040 modellinghe:0.035 15329:0.031 noncommunists:0.030 
    ==========================================================
    Cluster 97     tourney:0.361 eskimos:0.209 clubrajala:0.127 publicationthe:0.110 regionleblanc:0.063 
    ==========================================================
    Cluster 98     allpercussion:0.155 balletteacher:0.120 alfonsn:0.119 findon:0.090 venuesgrants:0.075 
    ==========================================================
    Cluster 99     wube:0.081 siveco:0.080 nedelchev:0.076 lowboy:0.076 shaftesburys:0.058 
    ==========================================================
    

The class of soccer (association football) players has been broken into two clusters (44 and 45). Same goes for Austrialian rules football players (clusters 26 and 48). The class of baseball players have been also broken into two clusters (16 and 91).

**A high value of K encourages pure clusters, but we cannot keep increasing K. For large enough K, related documents end up going to different clusters.**

That said, the result for K=100 is not entirely bad. After all, it gives us separate clusters for such categories as Brazil, wrestling, computer science and the Mormon Church. If we set K somewhere between 25 and 100, we should be able to avoid breaking up clusters while discovering new ones.

Also, we should ask ourselves how much **granularity** we want in our clustering. If we wanted a rough sketch of Wikipedia, we don't want too detailed clusters. On the other hand, having many clusters can be valuable when we are zooming into a certain part of Wikipedia.

**There is no golden rule for choosing K. It all depends on the particular application and domain we are in.**

Another heuristic people use that does not rely on so much visualization, which can be hard in many applications (including here!) is as follows.  Track heterogeneity versus K and look for the "elbow" of the curve where the heterogeneity decrease rapidly before this value of K, but then only gradually for larger values of K.  This naturally trades off between trying to minimize heterogeneity, but reduce model complexity.  In the heterogeneity versus K plot made above, we did not yet really see a flattening out of the heterogeneity, which might indicate that indeed K=100 is "reasonable" and we only see real overfitting for larger values of K (which are even harder to visualize using the methods we attempted above.)

**Quiz Question**. Another sign of too large K is having lots of small clusters. Look at the distribution of cluster sizes (by number of member data points). How many of the 100 clusters have fewer than 236 articles, i.e. 0.4% of the dataset?

Hint: Use `cluster_assignment[100]()`, with the extra pair of parentheses for delayed loading.


```python
np.sum(np.bincount(cluster_assignment[100]()) < 236)
```




    29



### Takeaway

Keep in mind though that tiny clusters aren't necessarily bad. A tiny cluster of documents that really look like each others is definitely preferable to a medium-sized cluster of documents with mixed content. However, having too few articles in a cluster may cause overfitting by reading too much into a limited pool of training data.
