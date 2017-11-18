
# Hierarchical Clustering

**Hierarchical clustering** refers to a class of clustering methods that seek to build a **hierarchy** of clusters, in which some clusters contain others. In this assignment, we will explore a top-down approach, recursively bipartitioning the data using k-means.

**Note to Amazon EC2 users**: To conserve memory, make sure to stop all the other notebooks before running this notebook.

## Import packages

The following code block will check if you have the correct version of GraphLab Create. Any version later than 1.8.5 will do. To upgrade, read [this page](https://turi.com/download/upgrade-graphlab-create.html).


```python
import sframe                                            # see below for install instruction
import matplotlib.pyplot as plt                                # plotting
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
%matplotlib inline
```

## Load the Wikipedia dataset


```python
wiki = sframe.SFrame('people_wiki.gl/')
```

    [INFO] sframe.cython.cy_server: SFrame v2.1 started. Logging C:\Users\SSQ\AppData\Local\Temp\sframe_server_1510989032.log.0
    


```python
wiki
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
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Harpdog_Brown&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Harpdog Brown</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">harpdog brown is a singer<br>and harmonica player who ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Franz_Rottensteiner&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Franz Rottensteiner</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">franz rottensteiner born<br>in waidmannsfeld lower ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/G-Enka&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">G-Enka</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">henry krvits born 30<br>december 1974 in tallinn ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Sam_Henderson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Sam Henderson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">sam henderson born<br>october 18 1969 is an ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Aaron_LaCrate&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Aaron LaCrate</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">aaron lacrate is an<br>american music producer ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Trevor_Ferguson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Trevor Ferguson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">trevor ferguson aka john<br>farrow born 11 november ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Grant_Nelson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Grant Nelson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">grant nelson born 27<br>april 1971 in london  ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Cathy_Caruth&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Cathy Caruth</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">cathy caruth born 1955 is<br>frank h t rhodes ...</td>
    </tr>
</table>
[59071 rows x 3 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



As we did in previous assignments, let's extract the TF-IDF features:

To run k-means on this dataset, we should convert the data matrix into a sparse matrix.


```python
def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']

    return csr_matrix( (data, indices, indptr), shape)

tf_idf = load_sparse_csr('people_wiki_tf_idf.npz')
map_index_to_word = sframe.SFrame('people_wiki_map_index_to_word.gl/')
```

To be consistent with the k-means assignment, let's normalize all vectors to have unit norm.


```python
tf_idf = normalize(tf_idf)
```


```python
tf_idf
```




    <59071x547979 sparse matrix of type '<type 'numpy.float64'>'
    	with 10379283 stored elements in Compressed Sparse Row format>




```python
map_index_to_word
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">feature</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">category</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">index</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">feature</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">bioarchaeologist</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">feature</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">leaguehockey</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">feature</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">electionruss</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">feature</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">teramoto</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">feature</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">trumpeterpercussionist</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">feature</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">spoofax</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">feature</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">mendelssohni</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">feature</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">crosswise</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">feature</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">yec</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">feature</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">asianthemed</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
    </tr>
</table>
[547979 rows x 3 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



## Bipartition the Wikipedia dataset using k-means

Recall our workflow for clustering text data with k-means:

1. Load the dataframe containing a dataset, such as the Wikipedia text dataset.
2. Extract the data matrix from the dataframe.
3. Run k-means on the data matrix with some value of k.
4. Visualize the clustering results using the centroids, cluster assignments, and the original dataframe. We keep the original dataframe around because the data matrix does not keep auxiliary information (in the case of the text dataset, the title of each article).

Let us modify the workflow to perform bipartitioning:

1. Load the dataframe containing a dataset, such as the Wikipedia text dataset.
2. Extract the data matrix from the dataframe.
3. Run k-means on the data matrix with k=2.
4. Divide the data matrix into two parts using the cluster assignments.
5. Divide the dataframe into two parts, again using the cluster assignments. This step is necessary to allow for visualization.
6. Visualize the bipartition of data.

We'd like to be able to repeat Steps 3-6 multiple times to produce a **hierarchy** of clusters such as the following:
```
                      (root)
                         |
            +------------+-------------+
            |                          |
         Cluster                    Cluster
     +------+-----+             +------+-----+
     |            |             |            |
   Cluster     Cluster       Cluster      Cluster
```
Each **parent cluster** is bipartitioned to produce two **child clusters**. At the very top is the **root cluster**, which consists of the entire dataset.

Now we write a wrapper function to bipartition a given cluster using k-means. There are three variables that together comprise the cluster:

* `dataframe`: a subset of the original dataframe that correspond to member rows of the cluster
* `matrix`: same set of rows, stored in sparse matrix format
* `centroid`: the centroid of the cluster (not applicable for the root cluster)

Rather than passing around the three variables separately, we package them into a Python dictionary. The wrapper function takes a single dictionary (representing a parent cluster) and returns two dictionaries (representing the child clusters).


```python
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
    data_matrix_left_child, data_matrix_right_child = data_matrix[cluster_assignment==0], \
                                                      data_matrix[cluster_assignment==1]
    
    # Divide the dataframe into two parts, again using the cluster assignments.
    cluster_assignment_sa = sframe.SArray(cluster_assignment) # minor format conversion
    dataframe_left_child, dataframe_right_child     = dataframe[cluster_assignment_sa==0], \
                                                      dataframe[cluster_assignment_sa==1]
        
    
    # Package relevant variables for the child clusters
    cluster_left_child  = {'matrix': data_matrix_left_child,
                           'dataframe': dataframe_left_child,
                           'centroid': centroids[0]}
    cluster_right_child = {'matrix': data_matrix_right_child,
                           'dataframe': dataframe_right_child,
                           'centroid': centroids[1]}
    
    return (cluster_left_child, cluster_right_child)
```

The following cell performs bipartitioning of the Wikipedia dataset. Allow 20-60 seconds to finish.

Note. For the purpose of the assignment, we set an explicit seed (`seed=1`) to produce identical outputs for every run. In pratical applications, you might want to use different random seeds for all runs.


```python
wiki_data = {'matrix': tf_idf, 'dataframe': wiki} # no 'centroid' for the root cluster
left_child, right_child = bipartition(wiki_data, maxiter=100, num_runs=6, seed=1)
```

Let's examine the contents of one of the two clusters, which we call the `left_child`, referring to the tree visualization above.


```python
left_child
```




    {'centroid': array([  0.00000000e+00,   8.57526623e-06,   0.00000000e+00, ...,
              1.38560691e-04,   6.46049863e-05,   2.26551103e-05]),
     'dataframe': Columns:
     	URI	str
     	name	str
     	text	str
     
     Rows: Unknown
     
     Data:
     +-------------------------------+-------------------------------+
     |              URI              |              name             |
     +-------------------------------+-------------------------------+
     | <http://dbpedia.org/resour... |         Digby Morrell         |
     | <http://dbpedia.org/resour... | Paddy Dunne (Gaelic footba... |
     | <http://dbpedia.org/resour... |         Ceiron Thomas         |
     | <http://dbpedia.org/resour... |          Adel Sellimi         |
     | <http://dbpedia.org/resour... |          Vic Stasiuk          |
     | <http://dbpedia.org/resour... |          Leon Hapgood         |
     | <http://dbpedia.org/resour... |           Dom Flora           |
     | <http://dbpedia.org/resour... |           Bob Reece           |
     | <http://dbpedia.org/resour... | Bob Adams (American football) |
     | <http://dbpedia.org/resour... |           Marc Logan          |
     +-------------------------------+-------------------------------+
     +-------------------------------+
     |              text             |
     +-------------------------------+
     | digby morrell born 10 octo... |
     | paddy dunne was a gaelic f... |
     | ceiron thomas born 23 octo... |
     | adel sellimi arabic was bo... |
     | victor john stasiuk born m... |
     | leon duane hapgood born 7 ... |
     | dominick a dom flora born ... |
     | robert scott reece born ja... |
     | robert bruce bob adams bor... |
     | marc anthony logan born ma... |
     +-------------------------------+
     [? rows x 3 columns]
     Note: Only the head of the SFrame is printed. This SFrame is lazily evaluated.
     You can use sf.materialize() to force materialization.,
     'matrix': <11510x547979 sparse matrix of type '<type 'numpy.float64'>'
     	with 1885831 stored elements in Compressed Sparse Row format>}



And here is the content of the other cluster we named `right_child`.


```python
right_child
```




    {'centroid': array([  3.00882137e-06,   0.00000000e+00,   2.88868244e-06, ...,
              1.10291526e-04,   9.00609890e-05,   2.03703564e-05]),
     'dataframe': Columns:
     	URI	str
     	name	str
     	text	str
     
     Rows: Unknown
     
     Data:
     +-------------------------------+---------------------+
     |              URI              |         name        |
     +-------------------------------+---------------------+
     | <http://dbpedia.org/resour... |    Alfred J. Lewy   |
     | <http://dbpedia.org/resour... |    Harpdog Brown    |
     | <http://dbpedia.org/resour... | Franz Rottensteiner |
     | <http://dbpedia.org/resour... |        G-Enka       |
     | <http://dbpedia.org/resour... |    Sam Henderson    |
     | <http://dbpedia.org/resour... |    Aaron LaCrate    |
     | <http://dbpedia.org/resour... |   Trevor Ferguson   |
     | <http://dbpedia.org/resour... |     Grant Nelson    |
     | <http://dbpedia.org/resour... |     Cathy Caruth    |
     | <http://dbpedia.org/resour... |     Sophie Crumb    |
     +-------------------------------+---------------------+
     +-------------------------------+
     |              text             |
     +-------------------------------+
     | alfred j lewy aka sandy le... |
     | harpdog brown is a singer ... |
     | franz rottensteiner born i... |
     | henry krvits born 30 decem... |
     | sam henderson born october... |
     | aaron lacrate is an americ... |
     | trevor ferguson aka john f... |
     | grant nelson born 27 april... |
     | cathy caruth born 1955 is ... |
     | sophia violet sophie crumb... |
     +-------------------------------+
     [? rows x 3 columns]
     Note: Only the head of the SFrame is printed. This SFrame is lazily evaluated.
     You can use sf.materialize() to force materialization.,
     'matrix': <47561x547979 sparse matrix of type '<type 'numpy.float64'>'
     	with 8493452 stored elements in Compressed Sparse Row format>}



## Visualize the bipartition

We provide you with a modified version of the visualization function from the k-means assignment. For each cluster, we print the top 5 words with highest TF-IDF weights in the centroid and display excerpts for the 8 nearest neighbors of the centroid.


```python
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
```

Let's visualize the two child clusters:


```python
display_single_tf_idf_cluster(left_child, map_index_to_word)
```

    league:0.040 season:0.036 team:0.029 football:0.029 played:0.028 
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
    
    


```python
display_single_tf_idf_cluster(right_child, map_index_to_word)
```

    she:0.025 her:0.017 music:0.012 he:0.011 university:0.011 
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
    
    

The left cluster consists of athletes, whereas the right cluster consists of non-athletes. So far, we have a single-level hierarchy consisting of two clusters, as follows:

```
                                           Wikipedia
                                               +
                                               |
                    +--------------------------+--------------------+
                    |                                               |
                    +                                               +
                 Athletes                                      Non-athletes
```

Is this hierarchy good enough? **When building a hierarchy of clusters, we must keep our particular application in mind.** For instance, we might want to build a **directory** for Wikipedia articles. A good directory would let you quickly narrow down your search to a small set of related articles. The categories of athletes and non-athletes are too general to facilitate efficient search. For this reason, we decide to build another level into our hierarchy of clusters with the goal of getting more specific cluster structure at the lower level. To that end, we subdivide both the `athletes` and `non-athletes` clusters.

## Perform recursive bipartitioning

### Cluster of athletes

To help identify the clusters we've built so far, let's give them easy-to-read aliases:


```python
athletes = left_child
non_athletes = right_child
```

Using the bipartition function, we produce two child clusters of the athlete cluster:


```python
# Bipartition the cluster of athletes
left_child_athletes, right_child_athletes = bipartition(athletes, maxiter=100, num_runs=6, seed=1)
```

The left child cluster mainly consists of baseball players:


```python
display_single_tf_idf_cluster(left_child_athletes, map_index_to_word)
```

    baseball:0.111 league:0.103 major:0.051 games:0.046 season:0.045 
    * Steve Springer                                     0.89344
      steven michael springer born february 11 1961 is an american former professional baseball 
      player who appeared in major league baseball as a third baseman and
    * Dave Ford                                          0.89598
      david alan ford born december 29 1956 is a former major league baseball pitcher for the ba
      ltimore orioles born in cleveland ohio ford attended lincolnwest
    * Todd Williams                                      0.89823
      todd michael williams born february 13 1971 in syracuse new york is a former major league 
      baseball relief pitcher he attended east syracuseminoa high school
    * Justin Knoedler                                    0.90097
      justin joseph knoedler born july 17 1980 in springfield illinois is a former major league 
      baseball catcherknoedler was originally drafted by the st louis cardinals
    * Kevin Nicholson (baseball)                         0.90607
      kevin ronald nicholson born march 29 1976 is a canadian baseball shortstop he played part 
      of the 2000 season for the san diego padres of
    * Joe Strong                                         0.90638
      joseph benjamin strong born september 9 1962 in fairfield california is a former major lea
      gue baseball pitcher who played for the florida marlins from 2000
    * James Baldwin (baseball)                           0.90674
      james j baldwin jr born july 15 1971 is a former major league baseball pitcher he batted a
      nd threw righthanded in his 11season career he
    * James Garcia                                       0.90729
      james robert garcia born february 3 1980 is an american former professional baseball pitch
      er who played in the san francisco giants minor league system as
    
    

On the other hand, the right child cluster is a mix of players in association football, Austrailian rules football and ice hockey:


```python
display_single_tf_idf_cluster(right_child_athletes, map_index_to_word)
```

    season:0.034 football:0.033 team:0.031 league:0.029 played:0.027 
    * Gord Sherven                                       0.95562
      gordon r sherven born august 21 1963 in gravelbourg saskatchewan and raised in mankota sas
      katchewan is a retired canadian professional ice hockey forward who played
    * Ashley Prescott                                    0.95656
      ashley prescott born 11 september 1972 is a former australian rules footballer he played w
      ith the richmond and fremantle football clubs in the afl between
    * Chris Day                                          0.95656
      christopher nicholas chris day born 28 july 1975 is an english professional footballer who
       plays as a goalkeeper for stevenageday started his career at tottenham
    * Jason Roberts (footballer)                         0.95658
      jason andre davis roberts mbe born 25 january 1978 is a former professional footballer and
       now a football punditborn in park royal london roberts was
    * Todd Curley                                        0.95743
      todd curley born 14 january 1973 is a former australian rules footballer who played for co
      llingwood and the western bulldogs in the australian football league
    * Tony Smith (footballer, born 1957)                 0.95801
      anthony tony smith born 20 february 1957 is a former footballer who played as a central de
      fender in the football league in the 1970s and
    * Sol Campbell                                       0.95802
      sulzeer jeremiah sol campbell born 18 september 1974 is a former england international foo
      tballer a central defender he had a 19year career playing in the
    * Richard Ambrose                                    0.95924
      richard ambrose born 10 june 1972 is a former australian rules footballer who played with 
      the sydney swans in the australian football league afl he
    
    

Our hierarchy of clusters now looks like this:
```
                                           Wikipedia
                                               +
                                               |
                    +--------------------------+--------------------+
                    |                                               |
                    +                                               +
                 Athletes                                      Non-athletes
                    +
                    |
        +-----------+--------+
        |                    |
        |            association football/
        +          Austrailian rules football/
     baseball             ice hockey
```

Should we keep subdividing the clusters? If so, which cluster should we subdivide? To answer this question, we again think about our application. Since we organize our directory by topics, it would be nice to have topics that are about as coarse as each other. For instance, if one cluster is about baseball, we expect some other clusters about football, basketball, volleyball, and so forth. That is, **we would like to achieve similar level of granularity for all clusters.**

Notice that the right child cluster is more coarse than the left child cluster. The right cluster posseses a greater variety of topics than the left (ice hockey/association football/Austrialian football vs. baseball). So the right child cluster should be subdivided further to produce finer child clusters.

Let's give the clusters aliases as well:


```python
baseball            = left_child_athletes
ice_hockey_football = right_child_athletes
```

### Cluster of ice hockey players and football players

In answering the following quiz question, take a look at the topics represented in the top documents (those closest to the centroid), as well as the list of words with highest TF-IDF weights.

Let us bipartition the cluster of ice hockey and football players.


```python
left_child_ihs, right_child_ihs = bipartition(ice_hockey_football, maxiter=100, num_runs=6, seed=1)
display_single_tf_idf_cluster(left_child_ihs, map_index_to_word)
display_single_tf_idf_cluster(right_child_ihs, map_index_to_word)
```

    football:0.048 season:0.043 league:0.041 played:0.036 coach:0.034 
    * Todd Curley                                        0.94578
      todd curley born 14 january 1973 is a former australian rules footballer who played for co
      llingwood and the western bulldogs in the australian football league
    * Tony Smith (footballer, born 1957)                 0.94606
      anthony tony smith born 20 february 1957 is a former footballer who played as a central de
      fender in the football league in the 1970s and
    * Chris Day                                          0.94623
      christopher nicholas chris day born 28 july 1975 is an english professional footballer who
       plays as a goalkeeper for stevenageday started his career at tottenham
    * Ashley Prescott                                    0.94632
      ashley prescott born 11 september 1972 is a former australian rules footballer he played w
      ith the richmond and fremantle football clubs in the afl between
    * Jason Roberts (footballer)                         0.94633
      jason andre davis roberts mbe born 25 january 1978 is a former professional footballer and
       now a football punditborn in park royal london roberts was
    * David Hamilton (footballer)                        0.94925
      david hamilton born 7 november 1960 is an english former professional association football
       player who played as a midfielder he won caps for the england
    * Richard Ambrose                                    0.94941
      richard ambrose born 10 june 1972 is a former australian rules footballer who played with 
      the sydney swans in the australian football league afl he
    * Neil Grayson                                       0.94958
      neil grayson born 1 november 1964 in york is an english footballer who last played as a st
      riker for sutton towngraysons first club was local
    
    championships:0.045 tour:0.043 championship:0.035 world:0.031 won:0.031 
    * Alessandra Aguilar                                 0.93856
      alessandra aguilar born 1 july 1978 in lugo is a spanish longdistance runner who specialis
      es in marathon running she represented her country in the event
    * Heather Samuel                                     0.93973
      heather barbara samuel born 6 july 1970 is a retired sprinter from antigua and barbuda who
       specialized in the 100 and 200 metres in 1990
    * Viola Kibiwot                                      0.94015
      viola jelagat kibiwot born december 22 1983 in keiyo district is a runner from kenya who s
      pecialises in the 1500 metres kibiwot won her first
    * Ayelech Worku                                      0.94031
      ayelech worku born june 12 1979 is an ethiopian longdistance runner most known for winning
       two world championships bronze medals on the 5000 metres she
    * Krisztina Papp                                     0.94077
      krisztina papp born 17 december 1982 in eger is a hungarian long distance runner she is th
      e national indoor record holder over 5000 mpapp began
    * Petra Lammert                                      0.94215
      petra lammert born 3 march 1984 in freudenstadt badenwrttemberg is a former german shot pu
      tter and current bobsledder she was the 2009 european indoor champion
    * Morhad Amdouni                                     0.94217
      morhad amdouni born 21 january 1988 in portovecchio is a french middle and longdistance ru
      nner he was european junior champion in track and cross country
    * Brian Davis (golfer)                               0.94369
      brian lester davis born 2 august 1974 is an english professional golferdavis was born in l
      ondon he turned professional in 1994 and became a member
    
    

**Quiz Question**. Which diagram best describes the hierarchy right after splitting the `ice_hockey_football` cluster? Refer to the quiz form for the diagrams.

**Caution**. The granularity criteria is an imperfect heuristic and must be taken with a grain of salt. It takes a lot of manual intervention to obtain a good hierarchy of clusters.

* **If a cluster is highly mixed, the top articles and words may not convey the full picture of the cluster.** Thus, we may be misled if we judge the purity of clusters solely by their top documents and words. 
* **Many interesting topics are hidden somewhere inside the clusters but do not appear in the visualization.** We may need to subdivide further to discover new topics. For instance, subdividing the `ice_hockey_football` cluster led to the appearance of runners and golfers.

### Cluster of non-athletes

Now let us subdivide the cluster of non-athletes.


```python
# Bipartition the cluster of non-athletes
left_child_non_athletes, right_child_non_athletes = bipartition(non_athletes, maxiter=100, num_runs=6, seed=1)
```


```python
display_single_tf_idf_cluster(left_child_non_athletes, map_index_to_word)
```

    he:0.013 music:0.012 university:0.011 film:0.010 his:0.009 
    * Wilson McLean                                      0.97870
      wilson mclean born 1937 is a scottish illustrator and artist he has illustrated primarily 
      in the field of advertising but has also provided cover art
    * Julian Knowles                                     0.97938
      julian knowles is an australian composer and performer specialising in new and emerging te
      chnologies his creative work spans the fields of composition for theatre dance
    * James A. Joseph                                    0.98042
      james a joseph born 1935 is an american former diplomatjoseph is professor of the practice
       of public policy studies at duke university and founder of
    * Barry Sullivan (lawyer)                            0.98054
      barry sullivan is a chicago lawyer and as of july 1 2009 the cooney conway chair in advoca
      cy at loyola university chicago school of law
    * Archie Brown                                       0.98081
      archibald haworth brown cmg fba commonly known as archie brown born 10 may 1938 is a briti
      sh political scientist and historian in 2005 he became
    * Michael Joseph Smith                               0.98124
      michael joseph smith is an american jazz and american classical composer and pianist born 
      in tiline kentucky he has worked extensively in europe and asia
    * Craig Pruess                                       0.98125
      craig pruess born 1950 is an american composer musician arranger and gold platinum record 
      producer who has been living in britain since 1973 his career
    * David J. Elliott                                   0.98128
      david elliott is professor of music and music education at new york universityelliott was 
      educated at the university of toronto bmus m mus and bed
    
    


```python
display_single_tf_idf_cluster(right_child_non_athletes, map_index_to_word)
```

    she:0.126 her:0.082 film:0.013 actress:0.012 music:0.012 
    * Janet Jackson                                      0.93808
      janet damita jo jackson born may 16 1966 is an american singer songwriter and actress know
      n for a series of sonically innovative socially conscious and
    * Lauren Royal                                       0.93867
      lauren royal born march 3 circa 1965 is a book writer from california royal has written bo
      th historic and novelistic booksa selfproclaimed angels baseball fan
    * Barbara Hershey                                    0.93941
      barbara hershey born barbara lynn herzstein february 5 1948 once known as barbara seagull 
      is an american actress in a career spanning nearly 50 years
    * Jane Fonda                                         0.94102
      jane fonda born lady jayne seymour fonda december 21 1937 is an american actress writer po
      litical activist former fashion model and fitness guru she is
    * Alexandra Potter                                   0.94190
      alexandra potter born 1970 is a british author of romantic comediesborn in bradford yorksh
      ire england and educated at liverpool university gaining an honors degree in
    * Janine Shepherd                                    0.94219
      janine lee shepherd am born 1962 is an australian pilot and former crosscountry skier shep
      herds career as an athlete ended when she suffered major injuries
    * Cher                                               0.94231
      cher r born cherilyn sarkisian may 20 1946 is an american singer actress and television ho
      st described as embodying female autonomy in a maledominated industry
    * Ellina Graypel                                     0.94233
      ellina graypel born july 19 1972 is an awardwinning russian singersongwriter she was born 
      near the volga river in the heart of russia she spent
    
    

Neither of the clusters show clear topics, apart from the genders. Let us divide them further.


```python
male_non_athletes = left_child_non_athletes
female_non_athletes = right_child_non_athletes
```

**Quiz Question**. Let us bipartition the clusters `male_non_athletes` and `female_non_athletes`. Which diagram best describes the resulting hierarchy of clusters for the non-athletes? Refer to the quiz for the diagrams.

**Note**. Use `maxiter=100, num_runs=6, seed=1` for consistency of output.


```python
left_child_male_non_athletes, right_child_male_non_athletes = bipartition(male_non_athletes, maxiter=100, num_runs=6, seed=1)
display_single_tf_idf_cluster(left_child_male_non_athletes, map_index_to_word)
display_single_tf_idf_cluster(right_child_male_non_athletes, map_index_to_word)
```

    university:0.017 he:0.015 law:0.013 served:0.013 research:0.013 
    * Barry Sullivan (lawyer)                            0.97075
      barry sullivan is a chicago lawyer and as of july 1 2009 the cooney conway chair in advoca
      cy at loyola university chicago school of law
    * James A. Joseph                                    0.97344
      james a joseph born 1935 is an american former diplomatjoseph is professor of the practice
       of public policy studies at duke university and founder of
    * David Anderson (British Columbia politician)       0.97383
      david a anderson pc oc born august 16 1937 in victoria british columbia is a former canadi
      an cabinet minister educated at victoria college in victoria
    * Sven Erik Holmes                                   0.97469
      sven erik holmes is a former federal judge and currently the vice chairman legal risk and 
      regulatory and chief legal officer for kpmg llp a
    * Andrew Fois                                        0.97558
      andrew fois is an attorney living and working in washington dc as of april 9 2012 he will 
      be serving as the deputy attorney general
    * William Robert Graham                              0.97564
      william robert graham born june 15 1937 was chairman of president reagans general advisory
       committee on arms control from 1982 to 1985 a deputy administrator
    * John C. Eastman                                    0.97585
      john c eastman born april 21 1960 is a conservative american law professor and constitutio
      nal law scholar he is the henry salvatori professor of law
    * M. Cherif Bassiouni                                0.97587
      mahmoud cherif bassiouni was born in cairo egypt in 1937 and immigrated to the united stat
      es in 1962 he is emeritus professor of law at
    
    music:0.023 film:0.020 album:0.014 band:0.014 art:0.013 
    * Julian Knowles                                     0.97192
      julian knowles is an australian composer and performer specialising in new and emerging te
      chnologies his creative work spans the fields of composition for theatre dance
    * Peter Combe                                        0.97292
      peter combe born 20 october 1948 is an australian childrens entertainer and musicianmusica
      l genre childrens musiche has had 22 releases including seven gold albums two
    * Craig Pruess                                       0.97346
      craig pruess born 1950 is an american composer musician arranger and gold platinum record 
      producer who has been living in britain since 1973 his career
    * Ceiri Torjussen                                    0.97420
      ceiri torjussen born 1976 is a composer who has contributed music to dozens of film and te
      levision productions in the ushis music was described by
    * Wilson McLean                                      0.97455
      wilson mclean born 1937 is a scottish illustrator and artist he has illustrated primarily 
      in the field of advertising but has also provided cover art
    * Brenton Broadstock                                 0.97471
      brenton broadstock ao born 1952 is an australian composerbroadstock was born in melbourne 
      he studied history politics and music at monash university and later composition
    * Michael Peter Smith                                0.97499
      michael peter smith born september 7 1941 is a chicagobased singersongwriter rolling stone
       magazine once called him the greatest songwriter in the english language he
    * Third Hawkins                                      0.97553
      born maurice hawkins third hawkins is a recognized music producer in and out of the dmv ar
      ea including his hometown of baltimore maryland he has
    
    


```python
left_child_female_non_athletes, right_child_female_non_athletes = bipartition(female_non_athletes, maxiter=100, num_runs=6, seed=1)
display_single_tf_idf_cluster(left_child_female_non_athletes, map_index_to_word)
display_single_tf_idf_cluster(right_child_female_non_athletes, map_index_to_word)
```

    she:0.121 her:0.100 actress:0.031 film:0.030 music:0.028 
    * Janet Jackson                                      0.92374
      janet damita jo jackson born may 16 1966 is an american singer songwriter and actress know
      n for a series of sonically innovative socially conscious and
    * Barbara Hershey                                    0.92524
      barbara hershey born barbara lynn herzstein february 5 1948 once known as barbara seagull 
      is an american actress in a career spanning nearly 50 years
    * Madonna (entertainer)                              0.92753
      madonna louise ciccone tkoni born august 16 1958 is an american singer songwriter actress 
      and businesswoman she achieved popularity by pushing the boundaries of lyrical
    * Cher                                               0.92909
      cher r born cherilyn sarkisian may 20 1946 is an american singer actress and television ho
      st described as embodying female autonomy in a maledominated industry
    * Candice Bergen                                     0.93266
      candice patricia bergen born may 9 1946 is an american actress and former fashion model fo
      r her role as the title character on the cbs
    * Glenn Close                                        0.93426
      glenn close born march 19 1947 is an american film television and stage actress throughout
       her long and varied career she has been consistently acclaimed
    * Jane Fonda                                         0.93515
      jane fonda born lady jayne seymour fonda december 21 1937 is an american actress writer po
      litical activist former fashion model and fitness guru she is
    * Judi Dench                                         0.93624
      dame judith olivia dench ch dbe frsa born 9 december 1934 is an english actress and author
       dench made her professional debut in 1957 with
    
    she:0.130 her:0.072 women:0.014 miss:0.014 university:0.013 
    * Lauren Royal                                       0.93939
      lauren royal born march 3 circa 1965 is a book writer from california royal has written bo
      th historic and novelistic booksa selfproclaimed angels baseball fan
    * %C3%81ine Hyland                                   0.93940
      ine hyland ne donlon is emeritus professor of education and former vicepresident of univer
      sity college cork ireland she was born in 1942 in athboy co
    * Dorothy E. Smith                                   0.94113
      dorothy edith smithborn july 6 1926 is a canadian sociologist with research interests besi
      des in sociology in many disciplines including womens studies psychology and educational
    * Kayee Griffin                                      0.94162
      kayee frances griffin born 6 february 1950 is an australian politician and former australi
      an labor party member of the new south wales legislative council serving
    * Janine Shepherd                                    0.94252
      janine lee shepherd am born 1962 is an australian pilot and former crosscountry skier shep
      herds career as an athlete ended when she suffered major injuries
    * Bhama Srinivasan                                   0.94281
      bhama srinivasan april 22 1935 is a mathematician known for her work in the representation
       theory of finite groups her contributions were honored with the
    * Ellen Christine Christiansen                       0.94395
      ellen christine christiansen born 10 december 1964 is a norwegian politician representing 
      the conservative party and formerly the progress partyborn in oslo she finished her
    * Elvira Vinogradova                                 0.94420
      elvira vinogradova russian born june 16 1934 is a russian tv editorelvira belenina russian
       was born in 1934 in fergana ussr she went to school
    
    


```python

```
