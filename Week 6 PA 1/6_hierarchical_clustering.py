
# coding: utf-8

# # Hierarchical Clustering

# **Hierarchical clustering** refers to a class of clustering methods that seek to build a **hierarchy** of clusters, in which some clusters contain others. In this assignment, we will explore a top-down approach, recursively bipartitioning the data using k-means.

# **Note to Amazon EC2 users**: To conserve memory, make sure to stop all the other notebooks before running this notebook.

# ## Import packages

# The following code block will check if you have the correct version of GraphLab Create. Any version later than 1.8.5 will do. To upgrade, read [this page](https://turi.com/download/upgrade-graphlab-create.html).

# In[1]:

import sframe                                            # see below for install instruction
import matplotlib.pyplot as plt                                # plotting
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
get_ipython().magic(u'matplotlib inline')


# ## Load the Wikipedia dataset

# In[2]:

wiki = sframe.SFrame('people_wiki.gl/')


# In[8]:

wiki


# As we did in previous assignments, let's extract the TF-IDF features:

# To run k-means on this dataset, we should convert the data matrix into a sparse matrix.

# In[3]:

def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']

    return csr_matrix( (data, indices, indptr), shape)

tf_idf = load_sparse_csr('people_wiki_tf_idf.npz')
map_index_to_word = sframe.SFrame('people_wiki_map_index_to_word.gl/')


# To be consistent with the k-means assignment, let's normalize all vectors to have unit norm.

# In[4]:

tf_idf = normalize(tf_idf)


# In[11]:

tf_idf


# In[15]:

map_index_to_word


# ## Bipartition the Wikipedia dataset using k-means

# Recall our workflow for clustering text data with k-means:
# 
# 1. Load the dataframe containing a dataset, such as the Wikipedia text dataset.
# 2. Extract the data matrix from the dataframe.
# 3. Run k-means on the data matrix with some value of k.
# 4. Visualize the clustering results using the centroids, cluster assignments, and the original dataframe. We keep the original dataframe around because the data matrix does not keep auxiliary information (in the case of the text dataset, the title of each article).
# 
# Let us modify the workflow to perform bipartitioning:
# 
# 1. Load the dataframe containing a dataset, such as the Wikipedia text dataset.
# 2. Extract the data matrix from the dataframe.
# 3. Run k-means on the data matrix with k=2.
# 4. Divide the data matrix into two parts using the cluster assignments.
# 5. Divide the dataframe into two parts, again using the cluster assignments. This step is necessary to allow for visualization.
# 6. Visualize the bipartition of data.
# 
# We'd like to be able to repeat Steps 3-6 multiple times to produce a **hierarchy** of clusters such as the following:
# ```
#                       (root)
#                          |
#             +------------+-------------+
#             |                          |
#          Cluster                    Cluster
#      +------+-----+             +------+-----+
#      |            |             |            |
#    Cluster     Cluster       Cluster      Cluster
# ```
# Each **parent cluster** is bipartitioned to produce two **child clusters**. At the very top is the **root cluster**, which consists of the entire dataset.
# 
# Now we write a wrapper function to bipartition a given cluster using k-means. There are three variables that together comprise the cluster:
# 
# * `dataframe`: a subset of the original dataframe that correspond to member rows of the cluster
# * `matrix`: same set of rows, stored in sparse matrix format
# * `centroid`: the centroid of the cluster (not applicable for the root cluster)
# 
# Rather than passing around the three variables separately, we package them into a Python dictionary. The wrapper function takes a single dictionary (representing a parent cluster) and returns two dictionaries (representing the child clusters).

# In[9]:

def bipartition(cluster, maxiter=400, num_runs=4, seed=None):
    '''cluster: should be a dictionary containing the following keys
                * dataframe: original dataframe
                * matrix:    same data, in matrix format
                * centroid:  centroid for this particular cluster'''
    
    data_matrix = cluster['matrix']
    dataframe   = cluster['dataframe']
    
    # Run k-means on the data matrix with k=2. We use scikit-learn here to simplify workflow.
    kmeans_model = KMeans(n_clusters=2, max_iter=maxiter, n_init=num_runs, random_state=seed, n_jobs=1)
    kmeans_model.fit(data_matrix)
    centroids, cluster_assignment = kmeans_model.cluster_centers_, kmeans_model.labels_
    
    # Divide the data matrix into two parts using the cluster assignments.
    data_matrix_left_child, data_matrix_right_child = data_matrix[cluster_assignment==0],                                                       data_matrix[cluster_assignment==1]
    
    # Divide the dataframe into two parts, again using the cluster assignments.
    cluster_assignment_sa = sframe.SArray(cluster_assignment) # minor format conversion
    dataframe_left_child, dataframe_right_child     = dataframe[cluster_assignment_sa==0],                                                       dataframe[cluster_assignment_sa==1]
        
    
    # Package relevant variables for the child clusters
    cluster_left_child  = {'matrix': data_matrix_left_child,
                           'dataframe': dataframe_left_child,
                           'centroid': centroids[0]}
    cluster_right_child = {'matrix': data_matrix_right_child,
                           'dataframe': dataframe_right_child,
                           'centroid': centroids[1]}
    
    return (cluster_left_child, cluster_right_child)


# The following cell performs bipartitioning of the Wikipedia dataset. Allow 20-60 seconds to finish.
# 
# Note. For the purpose of the assignment, we set an explicit seed (`seed=1`) to produce identical outputs for every run. In pratical applications, you might want to use different random seeds for all runs.

# In[12]:

wiki_data = {'matrix': tf_idf, 'dataframe': wiki} # no 'centroid' for the root cluster
left_child, right_child = bipartition(wiki_data, maxiter=100, num_runs=6, seed=1)


# Let's examine the contents of one of the two clusters, which we call the `left_child`, referring to the tree visualization above.

# In[13]:

left_child


# And here is the content of the other cluster we named `right_child`.

# In[14]:

right_child


# ## Visualize the bipartition

# We provide you with a modified version of the visualization function from the k-means assignment. For each cluster, we print the top 5 words with highest TF-IDF weights in the centroid and display excerpts for the 8 nearest neighbors of the centroid.

# In[16]:

def display_single_tf_idf_cluster(cluster, map_index_to_word):
    '''map_index_to_word: SFrame specifying the mapping betweeen words and column indices'''
    
    wiki_subset   = cluster['dataframe']
    tf_idf_subset = cluster['matrix']
    centroid      = cluster['centroid']
    
    # Print top 5 words with largest TF-IDF weights in the cluster
    idx = centroid.argsort()[::-1]
    for i in xrange(5):
        print('{0:s}:{1:.3f}'.format(map_index_to_word['category'][idx[i]], centroid[idx[i]])),
    print('')
    
    # Compute distances from the centroid to all data points in the cluster.
    distances = pairwise_distances(tf_idf_subset, [centroid], metric='euclidean').flatten()
    # compute nearest neighbors of the centroid within the cluster.
    nearest_neighbors = distances.argsort()
    # For 8 nearest neighbors, print the title as well as first 180 characters of text.
    # Wrap the text at 80-character mark.
    for i in xrange(8):
        text = ' '.join(wiki_subset[nearest_neighbors[i]]['text'].split(None, 25)[0:25])
        print('* {0:50s} {1:.5f}\n  {2:s}\n  {3:s}'.format(wiki_subset[nearest_neighbors[i]]['name'],
              distances[nearest_neighbors[i]], text[:90], text[90:180] if len(text) > 90 else ''))
    print('')


# Let's visualize the two child clusters:

# In[17]:

display_single_tf_idf_cluster(left_child, map_index_to_word)


# In[18]:

display_single_tf_idf_cluster(right_child, map_index_to_word)


# The left cluster consists of athletes, whereas the right cluster consists of non-athletes. So far, we have a single-level hierarchy consisting of two clusters, as follows:

# ```
#                                            Wikipedia
#                                                +
#                                                |
#                     +--------------------------+--------------------+
#                     |                                               |
#                     +                                               +
#                  Athletes                                      Non-athletes
# ```

# Is this hierarchy good enough? **When building a hierarchy of clusters, we must keep our particular application in mind.** For instance, we might want to build a **directory** for Wikipedia articles. A good directory would let you quickly narrow down your search to a small set of related articles. The categories of athletes and non-athletes are too general to facilitate efficient search. For this reason, we decide to build another level into our hierarchy of clusters with the goal of getting more specific cluster structure at the lower level. To that end, we subdivide both the `athletes` and `non-athletes` clusters.

# ## Perform recursive bipartitioning

# ### Cluster of athletes

# To help identify the clusters we've built so far, let's give them easy-to-read aliases:

# In[19]:

athletes = left_child
non_athletes = right_child


# Using the bipartition function, we produce two child clusters of the athlete cluster:

# In[20]:

# Bipartition the cluster of athletes
left_child_athletes, right_child_athletes = bipartition(athletes, maxiter=100, num_runs=6, seed=1)


# The left child cluster mainly consists of baseball players:

# In[21]:

display_single_tf_idf_cluster(left_child_athletes, map_index_to_word)


# On the other hand, the right child cluster is a mix of players in association football, Austrailian rules football and ice hockey:

# In[22]:

display_single_tf_idf_cluster(right_child_athletes, map_index_to_word)


# Our hierarchy of clusters now looks like this:
# ```
#                                            Wikipedia
#                                                +
#                                                |
#                     +--------------------------+--------------------+
#                     |                                               |
#                     +                                               +
#                  Athletes                                      Non-athletes
#                     +
#                     |
#         +-----------+--------+
#         |                    |
#         |            association football/
#         +          Austrailian rules football/
#      baseball             ice hockey
# ```

# Should we keep subdividing the clusters? If so, which cluster should we subdivide? To answer this question, we again think about our application. Since we organize our directory by topics, it would be nice to have topics that are about as coarse as each other. For instance, if one cluster is about baseball, we expect some other clusters about football, basketball, volleyball, and so forth. That is, **we would like to achieve similar level of granularity for all clusters.**
# 
# Notice that the right child cluster is more coarse than the left child cluster. The right cluster posseses a greater variety of topics than the left (ice hockey/association football/Austrialian football vs. baseball). So the right child cluster should be subdivided further to produce finer child clusters.

# Let's give the clusters aliases as well:

# In[23]:

baseball            = left_child_athletes
ice_hockey_football = right_child_athletes


# ### Cluster of ice hockey players and football players

# In answering the following quiz question, take a look at the topics represented in the top documents (those closest to the centroid), as well as the list of words with highest TF-IDF weights.
# 
# Let us bipartition the cluster of ice hockey and football players.

# In[24]:

left_child_ihs, right_child_ihs = bipartition(ice_hockey_football, maxiter=100, num_runs=6, seed=1)
display_single_tf_idf_cluster(left_child_ihs, map_index_to_word)
display_single_tf_idf_cluster(right_child_ihs, map_index_to_word)


# **Quiz Question**. Which diagram best describes the hierarchy right after splitting the `ice_hockey_football` cluster? Refer to the quiz form for the diagrams.

# **Caution**. The granularity criteria is an imperfect heuristic and must be taken with a grain of salt. It takes a lot of manual intervention to obtain a good hierarchy of clusters.
# 
# * **If a cluster is highly mixed, the top articles and words may not convey the full picture of the cluster.** Thus, we may be misled if we judge the purity of clusters solely by their top documents and words. 
# * **Many interesting topics are hidden somewhere inside the clusters but do not appear in the visualization.** We may need to subdivide further to discover new topics. For instance, subdividing the `ice_hockey_football` cluster led to the appearance of runners and golfers.

# ### Cluster of non-athletes

# Now let us subdivide the cluster of non-athletes.

# In[25]:

# Bipartition the cluster of non-athletes
left_child_non_athletes, right_child_non_athletes = bipartition(non_athletes, maxiter=100, num_runs=6, seed=1)


# In[26]:

display_single_tf_idf_cluster(left_child_non_athletes, map_index_to_word)


# In[27]:

display_single_tf_idf_cluster(right_child_non_athletes, map_index_to_word)


# Neither of the clusters show clear topics, apart from the genders. Let us divide them further.

# In[28]:

male_non_athletes = left_child_non_athletes
female_non_athletes = right_child_non_athletes


# **Quiz Question**. Let us bipartition the clusters `male_non_athletes` and `female_non_athletes`. Which diagram best describes the resulting hierarchy of clusters for the non-athletes? Refer to the quiz for the diagrams.
# 
# **Note**. Use `maxiter=100, num_runs=6, seed=1` for consistency of output.

# In[29]:

left_child_male_non_athletes, right_child_male_non_athletes = bipartition(male_non_athletes, maxiter=100, num_runs=6, seed=1)
display_single_tf_idf_cluster(left_child_male_non_athletes, map_index_to_word)
display_single_tf_idf_cluster(right_child_male_non_athletes, map_index_to_word)


# In[30]:

left_child_female_non_athletes, right_child_female_non_athletes = bipartition(female_non_athletes, maxiter=100, num_runs=6, seed=1)
display_single_tf_idf_cluster(left_child_female_non_athletes, map_index_to_word)
display_single_tf_idf_cluster(right_child_female_non_athletes, map_index_to_word)


# In[ ]:



