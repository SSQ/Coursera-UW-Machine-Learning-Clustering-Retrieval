
# Nearest Neighbors

When exploring a large set of documents -- such as Wikipedia, news articles, StackOverflow, etc. -- it can be useful to get a list of related material. To find relevant documents you typically
* Decide on a notion of similarity
* Find the documents that are most similar 

In the assignment you will
* Gain intuition for different notions of similarity and practice finding similar documents. 
* Explore the tradeoffs with representing documents using raw word counts and TF-IDF
* Explore the behavior of different distance metrics by looking at the Wikipedia pages most similar to President Obama’s page.

**Note to Amazon EC2 users**: To conserve memory, make sure to stop all the other notebooks before running this notebook.

## Import necessary packages

As usual we need to first import the Python packages that we will need.

The following code block will check if you have the correct version of GraphLab Create. Any version later than 1.8.5 will do. To upgrade, read [this page](https://turi.com/download/upgrade-graphlab-create.html).


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.sparse import csr_matrix      # sparse matrices
%matplotlib inline

```

## Load Wikipedia dataset

We will be using the same dataset of Wikipedia pages that we used in the Machine Learning Foundations course (Course 1). Each element of the dataset consists of a link to the wikipedia article, the name of the person, and the text of the article (in lowercase).  


```python
wiki = pd.read_csv('people_wiki.csv')
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
wiki.shape
```




    (59071, 3)



## Extract word count vectors

As we have seen in Course 1, we can extract word count vectors using a GraphLab utility function.  We add this as a column in `wiki`.


```python
def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']
    print data[0],indices[0], indptr[0],shape[0]
    return csr_matrix( (data, indices, indptr), shape)
```


```python
word_count = load_sparse_csr('people_wiki_word_count.npz')
```

    1 5877 0 59071
    


```python
word_count
```




    <59071x547979 sparse matrix of type '<type 'numpy.int64'>'
    	with 10379283 stored elements in Compressed Sparse Row format>




```python
type(word_count)
```




    scipy.sparse.csr.csr_matrix




```python
print word_count[0]
```

      (0, 5877)	1
      (0, 92219)	1
      (0, 227191)	1
      (0, 446948)	1
      (0, 468870)	1
      (0, 477285)	5
      (0, 492466)	1
      (0, 509506)	1
      (0, 514262)	1
      (0, 523996)	1
      (0, 528953)	1
      (0, 529843)	1
      (0, 533540)	1
      (0, 535034)	3
      (0, 535475)	1
      (0, 538022)	1
      (0, 538168)	1
      (0, 540827)	1
      (0, 541501)	1
      (0, 541760)	1
      (0, 542488)	1
      (0, 542854)	1
      (0, 542859)	1
      (0, 542919)	1
      (0, 543517)	2
      :	:
      (0, 547931)	1
      (0, 547934)	1
      (0, 547935)	1
      (0, 547938)	1
      (0, 547952)	1
      (0, 547956)	1
      (0, 547958)	1
      (0, 547959)	1
      (0, 547960)	1
      (0, 547962)	2
      (0, 547963)	1
      (0, 547964)	3
      (0, 547965)	4
      (0, 547966)	6
      (0, 547967)	5
      (0, 547969)	2
      (0, 547970)	5
      (0, 547971)	4
      (0, 547972)	5
      (0, 547973)	1
      (0, 547974)	4
      (0, 547975)	4
      (0, 547976)	13
      (0, 547977)	4
      (0, 547978)	27
    


```python
with open('people_wiki_map_index_to_word.json') as people_wiki_map_index_to_word:    
    map_index_to_word = json.load(people_wiki_map_index_to_word)
```


```python
len(map_index_to_word)
```




    547979




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

```

## Find nearest neighbors

Let's start by finding the nearest neighbors of the Barack Obama page using the word count vectors to represent the articles and Euclidean distance to measure distance.  For this, again will we use a GraphLab Create implementation of nearest neighbor search.


```python
from sklearn.neighbors import NearestNeighbors

model = NearestNeighbors(metric='euclidean', algorithm='brute')
model.fit(word_count)
```




    NearestNeighbors(algorithm='brute', leaf_size=30, metric='euclidean',
             metric_params=None, n_jobs=1, n_neighbors=5, p=2, radius=1.0)




```python
wiki[wiki['name'] == 'Barack Obama']
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
      <th>35817</th>
      <td>&lt;http://dbpedia.org/resource/Barack_Obama&gt;</td>
      <td>Barack Obama</td>
      <td>barack hussein obama ii brk husen bm born augu...</td>
    </tr>
  </tbody>
</table>
</div>




```python
distances, indices = model.kneighbors(word_count[35817], n_neighbors=10) # 1st arg: word count vector
```


```python
neighbors = pd.DataFrame(data={'distance':distances.flatten()},index=indices.flatten())
print wiki.join(neighbors).sort('distance')[['name','distance']][0:10]
```

                                 name   distance
    35817                Barack Obama   0.000000
    24478                   Joe Biden  33.075671
    28447              George W. Bush  34.394767
    35357            Lawrence Summers  36.152455
    14754                 Mitt Romney  36.166283
    13229            Francisco Barrio  36.331804
    31423              Walter Mondale  36.400549
    22745  Wynn Normington Hugh-Jones  36.496575
    36364                  Don Bonker  36.633318
    9210                 Andy Anstett  36.959437
    

    C:\Users\SSQ\AppData\Roaming\Python\Python27\site-packages\ipykernel\__main__.py:2: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=.....)
      from ipykernel import kernelapp as app
    


```python
neighbors.head(2)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35817</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>24478</th>
      <td>33.075671</td>
    </tr>
  </tbody>
</table>
</div>



Let's look at the top 10 nearest neighbors by performing the following query:


```python
# model.query(wiki[wiki['name']=='Barack Obama'], label='name', k=10)
```

All of the 10 people are politicians, but about half of them have rather tenuous connections with Obama, other than the fact that they are politicians.

* Francisco Barrio is a Mexican politician, and a former governor of Chihuahua.
* Walter Mondale and Don Bonker are Democrats who made their career in late 1970s.
* Wynn Normington Hugh-Jones is a former British diplomat and Liberal Party official.
* Andy Anstett is a former politician in Manitoba, Canada.

Nearest neighbors with raw word counts got some things right, showing all politicians in the query result, but missed finer and important details.

For instance, let's find out why Francisco Barrio was considered a close neighbor of Obama.  To do this, let's look at the most frequently used words in each of Barack Obama and Francisco Barrio's pages:


```python
def unpack_dict(matrix, map_index_to_word):
    #table = list(map_index_to_word.sort('index')['category'])
    # if you're not using SFrame, replace this line with
    table = sorted(map_index_to_word, key=map_index_to_word.get)
    
    
    data = matrix.data
    indices = matrix.indices
    indptr = matrix.indptr
    
    num_doc = matrix.shape[0]

    return [{k:v for k,v in zip([table[word_id] for word_id in indices[indptr[i]:indptr[i+1]] ],
                                 data[indptr[i]:indptr[i+1]].tolist())} \
               for i in xrange(num_doc) ]

wiki['word_count'] = unpack_dict(word_count, map_index_to_word)
```


```python
wiki.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>URI</th>
      <th>name</th>
      <th>text</th>
      <th>word_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>&lt;http://dbpedia.org/resource/Digby_Morrell&gt;</td>
      <td>Digby Morrell</td>
      <td>digby morrell born 10 october 1979 is a former...</td>
      <td>{u'selection': 1, u'carltons': 1, u'being': 1,...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>&lt;http://dbpedia.org/resource/Alfred_J._Lewy&gt;</td>
      <td>Alfred J. Lewy</td>
      <td>alfred j lewy aka sandy lewy graduated from un...</td>
      <td>{u'precise': 1, u'thomas': 1, u'they': 1, u'di...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>&lt;http://dbpedia.org/resource/Harpdog_Brown&gt;</td>
      <td>Harpdog Brown</td>
      <td>harpdog brown is a singer and harmonica player...</td>
      <td>{u'just': 1, u'issued': 1, u'mainly': 1, u'nom...</td>
    </tr>
  </tbody>
</table>
</div>




```python
"""
name = 'Barack Obama'
row = wiki[wiki['name'] == name]
dic = row['word_count'].iloc[0]
word_table = sorted(dic, key=dic.get, reverse=True)
print word_table
value_table = [dic[i] for i in word_table]
print value_table
print zip(word_table,value_table)
"""
```




    "\nname = 'Barack Obama'\nrow = wiki[wiki['name'] == name]\ndic = row['word_count'].iloc[0]\nword_table = sorted(dic, key=dic.get, reverse=True)\nprint word_table\nvalue_table = [dic[i] for i in word_table]\nprint value_table\nprint zip(word_table,value_table)\n"




```python
name = 'Barack Obama'
row = wiki[wiki['name'] == name]
dic = row['word_count'].iloc[0]
#print dic
word_count_ = pd.DataFrame(dic.items(), columns=['word','count'], index=None)
print word_count_.head(4)
```

               word  count
    0    operations      1
    1     represent      1
    2        office      2
    3  unemployment      1
    


```python
def top_words(name):
    """
    Get a table of the most frequent words in the given person's wikipedia page.
    """
    row = wiki[wiki['name'] == name]
    dic = row['word_count'].iloc[0]
    word_count_ = pd.DataFrame(dic.items(), columns=['word','count'])
    word_count_table = word_count_.sort(['count'], ascending=False)
    #word_count_table = word_count_table.fillna(0)
    #word_count_table = row[['word_count']].stack('word_count', new_column_name=['word','count'])
    return word_count_table #word_count_table.sort('count', ascending=False)

obama_words = top_words('Barack Obama')
print obama_words.head(10)
print '\n'
barrio_words = top_words('Francisco Barrio')
print barrio_words.head(10)
```

          word  count
    23     the     40
    215     in     30
    138    and     21
    127     of     18
    161     to     14
    37     his     11
    189  obama      9
    133    act      8
    156     he      7
    260      a      7
    
    
              word  count
    220        the     36
    84          of     24
    206        and     18
    184         in     17
    125         he     10
    17          to      9
    14   chihuahua      7
    159   governor      6
    211          a      6
    181         as      5
    

    C:\Users\SSQ\AppData\Roaming\Python\Python27\site-packages\ipykernel\__main__.py:8: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=.....)
    

Let's extract the list of most frequent words that appear in both Obama's and Barrio's documents. We've so far sorted all words from Obama and Barrio's articles by their word frequencies. We will now use a dataframe operation known as **join**. The **join** operation is very useful when it comes to playing around with data: it lets you combine the content of two tables using a shared column (in this case, the word column). See [the documentation](https://dato.com/products/create/docs/generated/graphlab.SFrame.join.html) for more details.

For instance, running
```
obama_words.join(barrio_words, on='word')
```
will extract the rows from both tables that correspond to the common words.


```python
combined_words = obama_words.set_index('word').join(barrio_words.set_index('word'), lsuffix='_obama', rsuffix='_barrio')
combined_words.head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count_obama</th>
      <th>count_barrio</th>
    </tr>
    <tr>
      <th>word</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>the</th>
      <td>40</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>in</th>
      <td>30</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>and</th>
      <td>21</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>of</th>
      <td>18</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>to</th>
      <td>14</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>his</th>
      <td>11</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>obama</th>
      <td>9</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>act</th>
      <td>8</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>he</th>
      <td>7</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>a</th>
      <td>7</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
"""
combined_words = obama_words.join(barrio_words, how='inner', on='word')
combined_words
"""
```




    "\ncombined_words = obama_words.join(barrio_words, how='inner', on='word')\ncombined_words\n"



Since both tables contained the column named `count`, SFrame automatically renamed one of them to prevent confusion. Let's rename the columns to tell which one is for which. By inspection, we see that the first column (`count`) is for Obama and the second (`count.1`) for Barrio.


```python
combined_words = combined_words.rename(index=str, columns={'count_obama':'Obama', 'count_barrio':'Barrio'})
combined_words.head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Obama</th>
      <th>Barrio</th>
    </tr>
    <tr>
      <th>word</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>the</th>
      <td>40</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>in</th>
      <td>30</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>and</th>
      <td>21</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>of</th>
      <td>18</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>to</th>
      <td>14</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>his</th>
      <td>11</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>obama</th>
      <td>9</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>act</th>
      <td>8</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>he</th>
      <td>7</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>a</th>
      <td>7</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

**Note**. The **join** operation does not enforce any particular ordering on the shared column. So to obtain, say, the five common words that appear most often in Obama's article, sort the combined table by the Obama column. Don't forget `ascending=False` to display largest counts first.


```python
combined_words.sort(columns='Obama', ascending=False)
combined_words.head(10)
```

    C:\Users\SSQ\AppData\Roaming\Python\Python27\site-packages\ipykernel\__main__.py:1: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=.....)
      if __name__ == '__main__':
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Obama</th>
      <th>Barrio</th>
    </tr>
    <tr>
      <th>word</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>the</th>
      <td>40</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>in</th>
      <td>30</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>and</th>
      <td>21</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>of</th>
      <td>18</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>to</th>
      <td>14</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>his</th>
      <td>11</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>obama</th>
      <td>9</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>act</th>
      <td>8</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>he</th>
      <td>7</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>a</th>
      <td>7</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>



**Quiz Question**. Among the words that appear in both Barack Obama and Francisco Barrio, take the 5 that appear most frequently in Obama. How many of the articles in the Wikipedia dataset contain all of those 5 words?

Hint:
* Refer to the previous paragraph for finding the words that appear in both articles. Sort the common words by their frequencies in Obama's article and take the largest five.
* Each word count vector is a Python dictionary. For each word count vector in SFrame, you'd have to check if the set of the 5 common words is a subset of the keys of the word count vector. Complete the function `has_top_words` to accomplish the task.
  - Convert the list of top 5 words into set using the syntax
```
set(common_words)
```
    where `common_words` is a Python list. See [this link](https://docs.python.org/2/library/stdtypes.html#set) if you're curious about Python sets.
  - Extract the list of keys of the word count dictionary by calling the [`keys()` method](https://docs.python.org/2/library/stdtypes.html#dict.keys).
  - Convert the list of keys into a set as well.
  - Use [`issubset()` method](https://docs.python.org/2/library/stdtypes.html#set) to check if all 5 words are among the keys.
* Now apply the `has_top_words` function on every row of the SFrame.
* Compute the sum of the result column to obtain the number of articles containing all the 5 top words.


```python
common_words = set(['the', 'in', 'and', 'of', 'to'])  # YOUR CODE HERE

def has_top_words(word_count_vector):
    # extract the keys of word_count_vector and convert it to a set
    unique_words = set(word_count_vector.keys())   # YOUR CODE HERE
    # return True if common_words is a subset of unique_words
    # return False otherwise
    #print unique_words
    return common_words.issubset(unique_words)  # YOUR CODE HERE

wiki['has_top_words'] = wiki['word_count'].apply(has_top_words)

# use has_top_words column to answer the quiz question
wiki.head(10) # YOUR CODE HERE
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>URI</th>
      <th>name</th>
      <th>text</th>
      <th>word_count</th>
      <th>has_top_words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>&lt;http://dbpedia.org/resource/Digby_Morrell&gt;</td>
      <td>Digby Morrell</td>
      <td>digby morrell born 10 october 1979 is a former...</td>
      <td>{u'selection': 1, u'carltons': 1, u'being': 1,...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>&lt;http://dbpedia.org/resource/Alfred_J._Lewy&gt;</td>
      <td>Alfred J. Lewy</td>
      <td>alfred j lewy aka sandy lewy graduated from un...</td>
      <td>{u'precise': 1, u'thomas': 1, u'they': 1, u'di...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>&lt;http://dbpedia.org/resource/Harpdog_Brown&gt;</td>
      <td>Harpdog Brown</td>
      <td>harpdog brown is a singer and harmonica player...</td>
      <td>{u'just': 1, u'issued': 1, u'mainly': 1, u'nom...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>&lt;http://dbpedia.org/resource/Franz_Rottensteiner&gt;</td>
      <td>Franz Rottensteiner</td>
      <td>franz rottensteiner born in waidmannsfeld lowe...</td>
      <td>{u'all': 1, u'bauforschung': 1, u'just': 1, u'...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>&lt;http://dbpedia.org/resource/G-Enka&gt;</td>
      <td>G-Enka</td>
      <td>henry krvits born 30 december 1974 in tallinn ...</td>
      <td>{u'legendary': 1, u'gangstergenka': 1, u'legen...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>&lt;http://dbpedia.org/resource/Sam_Henderson&gt;</td>
      <td>Sam Henderson</td>
      <td>sam henderson born october 18 1969 is an ameri...</td>
      <td>{u'now': 1, u'since': 1, u'less': 1, u'being':...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>&lt;http://dbpedia.org/resource/Aaron_LaCrate&gt;</td>
      <td>Aaron LaCrate</td>
      <td>aaron lacrate is an american music producer re...</td>
      <td>{u'exclusive': 2, u'played': 1, u'producer': 1...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>&lt;http://dbpedia.org/resource/Trevor_Ferguson&gt;</td>
      <td>Trevor Ferguson</td>
      <td>trevor ferguson aka john farrow born 11 novemb...</td>
      <td>{u'taxi': 1, u'salon': 1, u'gangs': 1, u'being...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>&lt;http://dbpedia.org/resource/Grant_Nelson&gt;</td>
      <td>Grant Nelson</td>
      <td>grant nelson born 27 april 1971 in london also...</td>
      <td>{u'houston': 1, u'frankie': 1, u'labels': 1, u...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>&lt;http://dbpedia.org/resource/Cathy_Caruth&gt;</td>
      <td>Cathy Caruth</td>
      <td>cathy caruth born 1955 is frank h t rhodes pro...</td>
      <td>{u'phenomenon': 1, u'deborash': 1, u'innovativ...</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(wiki[wiki['has_top_words'] == True])
```




    56066



**Checkpoint**. Check your `has_top_words` function on two random articles:


```python
set(['the']).issubset(set(wiki.iloc[32]['word_count'].keys()))
```




    True




```python
wiki.iloc[32]['word_count']['to']
```




    9L




```python
common_words = set(['the', 'in', 'and', 'of', 'to'])
for i in common_words:
    print i, set([i]).issubset(set(wiki.iloc[32]['word_count'].keys()))
```

    and True
    of True
    the True
    to True
    in True
    


```python
print 'Output from your function:', has_top_words(wiki.iloc[32]['word_count'])
print 'Correct output: True'
print 'Also check the length of unique_words. It should be 167'
```

    Output from your function: True
    Correct output: True
    Also check the length of unique_words. It should be 167
    


```python
print 'Output from your function:', has_top_words(wiki.iloc[33]['word_count'])
print 'Correct output: False'
print 'Also check the length of unique_words. It should be 188'
```

    Output from your function: False
    Correct output: False
    Also check the length of unique_words. It should be 188
    

**Quiz Question**. Measure the pairwise distance between the Wikipedia pages of Barack Obama, George W. Bush, and Joe Biden. Which of the three pairs has the smallest distance?

Hint: For this question, take the row vectors from the word count matrix that correspond to Obama, Bush, and Biden. To compute the Euclidean distance between any two sparse vectors, use [sklearn.metrics.pairwise.euclidean_distances](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html).


```python
print wiki.index[wiki['name'] == 'Barack Obama'].tolist()

```




    [35817]




```python
print wiki.index[wiki['name'] == 'George W. Bush'].tolist()
```




    [28447]




```python
print wiki.index[wiki['name'] == 'Joe Biden'].tolist()
```




    [24478]




```python
from sklearn.metrics.pairwise import euclidean_distances
print euclidean_distances(word_count[wiki.index[wiki['name'] == 'Barack Obama'].tolist()], word_count[wiki.index[wiki['name'] == 'George W. Bush'].tolist()])
print euclidean_distances(word_count[wiki.index[wiki['name'] == 'Barack Obama'].tolist()], word_count[wiki.index[wiki['name'] == 'Joe Biden'].tolist()])
print euclidean_distances(word_count[wiki.index[wiki['name'] == 'George W. Bush'].tolist()], word_count[wiki.index[wiki['name'] == 'Joe Biden'].tolist()])
```

    [[ 34.39476704]]
    [[ 33.07567082]]
    [[ 32.75667871]]
    

**Quiz Question**. Collect all words that appear both in Barack Obama and George W. Bush pages.  Out of those words, find the 10 words that show up most often in Obama's page. 


```python
obama_words = top_words('Barack Obama')
bush_words = top_words('George W. Bush')

obama_bush = obama_words.set_index('word').join(bush_words.set_index('word'), lsuffix='_obama', rsuffix='_bush')
obama_bush = obama_bush.rename(index=str, columns={'count_obama':'Obama', 'count_bush':'Bush'})
obama_bush.sort(columns='Obama', ascending=False)
dropped = obama_bush.dropna()
print dropped.head(10)
```

          Obama  Bush
    word             
    the      40  39.0
    in       30  22.0
    and      21  14.0
    of       18  14.0
    to       14  11.0
    his      11   6.0
    act       8   3.0
    he        7   8.0
    a         7   6.0
    as        6   6.0
    

    C:\Users\SSQ\AppData\Roaming\Python\Python27\site-packages\ipykernel\__main__.py:8: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=.....)
    C:\Users\SSQ\AppData\Roaming\Python\Python27\site-packages\ipykernel\__main__.py:6: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=.....)
    


```python

```

**Note.** Even though common words are swamping out important subtle differences, commonalities in rarer political words still matter on the margin. This is why politicians are being listed in the query result instead of musicians, for example. In the next subsection, we will introduce a different metric that will place greater emphasis on those rarer words.

## TF-IDF to the rescue

Much of the perceived commonalities between Obama and Barrio were due to occurrences of extremely frequent words, such as "the", "and", and "his". So nearest neighbors is recommending plausible results sometimes for the wrong reasons. 

To retrieve articles that are more relevant, we should focus more on rare words that don't happen in every article. **TF-IDF** (term frequency–inverse document frequency) is a feature representation that penalizes words that are too common.  Let's use GraphLab Create's implementation of TF-IDF and repeat the search for the 10 nearest neighbors of Barack Obama:


```python
#wiki['tf_idf'] = graphlab.text_analytics.tf_idf(wiki['word_count'])
tf_idf = load_sparse_csr('people_wiki_tf_idf.npz')
wiki['tf_idf'] = unpack_dict(tf_idf, map_index_to_word)
```

    10.9864953892 5877 0 59071
    


```python
model_tf_idf = NearestNeighbors(metric='euclidean', algorithm='brute')
model_tf_idf.fit(tf_idf)
```




    NearestNeighbors(algorithm='brute', leaf_size=30, metric='euclidean',
             metric_params=None, n_jobs=1, n_neighbors=5, p=2, radius=1.0)




```python
distances, indices = model_tf_idf.kneighbors(tf_idf[35817], n_neighbors=10)
```


```python
neighbors = pd.DataFrame(data={'distance':distances.flatten()}, index=indices.flatten())

print wiki.join(neighbors).sort('distance')[['name','distance']][0:10]
```

                              name    distance
    35817             Barack Obama    0.000000
    7914             Phil Schiliro  106.861014
    46811            Jeff Sessions  108.871674
    44681   Jesse Lee (politician)  109.045698
    38376           Samantha Power  109.108106
    6507              Bob Menendez  109.781867
    38714  Eric Stern (politician)  109.957788
    44825           James A. Guest  110.413889
    44368     Roland Grossenbacher  110.470609
    33417            Tulsi Gabbard  110.696998
    

    C:\Users\SSQ\AppData\Roaming\Python\Python27\site-packages\ipykernel\__main__.py:3: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=.....)
      app.launch_new_instance()
    


```python
#model_tf_idf.query(wiki[wiki['name'] == 'Barack Obama'], label='name', k=10)
```

Let's determine whether this list makes sense.
* With a notable exception of Roland Grossenbacher, the other 8 are all American politicians who are contemporaries of Barack Obama.
* Phil Schiliro, Jesse Lee, Samantha Power, and Eric Stern worked for Obama.

Clearly, the results are more plausible with the use of TF-IDF. Let's take a look at the word vector for Obama and Schilirio's pages. Notice that TF-IDF representation assigns a weight to each word. This weight captures relative importance of that word in the document. Let us sort the words in Obama's article by their TF-IDF weights; we do the same for Schiliro's article as well.


```python
def top_words_tf_idf(name):
    row = wiki[wiki['name'] == name]
    row = wiki[wiki['name'] == name]
    dic = row['tf_idf'].iloc[0]
    word_weight_ = pd.DataFrame(dic.items(), columns=['word','weight'])
    word_weight_table = word_weight_.sort(['weight'], ascending=False)
    #word_count_table = row[['word_count']].stack('word_count', new_column_name=['word','count'])
    return word_weight_table #word_count_table.sort('count', ascending=False)

obama_tf_idf = top_words_tf_idf('Barack Obama')
print obama_tf_idf.head(10)

schiliro_tf_idf = top_words_tf_idf('Phil Schiliro')
print schiliro_tf_idf.head(10)
```

                word     weight
    189        obama  43.295653
    133          act  27.678223
    157         iraq  17.747379
    208      control  14.887061
    99           law  14.722936
    210      ordered  14.533374
    134     military  13.115933
    254  involvement  12.784385
    259     response  12.784385
    270   democratic  12.410689
                    word     weight
    16          schiliro  21.972991
    8              staff  15.856442
    87     congressional  13.547088
    104  daschleschiliro  10.986495
    82             obama   9.621256
    28            waxman   9.040585
    106        president   9.033587
    24          2014from   8.683910
    107              law   7.361468
    84        consultant   6.913104
    

    C:\Users\SSQ\AppData\Roaming\Python\Python27\site-packages\ipykernel\__main__.py:6: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=.....)
    

Using the **join** operation we learned earlier, try your hands at computing the common words shared by Obama's and Schiliro's articles. Sort the common words by their TF-IDF weights in Obama's document.


```python
obama_words = top_words_tf_idf('Barack Obama')
schiliro_words = top_words_tf_idf('Phil Schiliro')

obama_bush = obama_words.set_index('word').join(schiliro_words.set_index('word'), lsuffix='_obama', rsuffix='_schiliro')
obama_bush = obama_bush.rename(index=str, columns={'weight_obama':'Obama', 'weight_schiliro':'Schiliro'})
obama_bush.sort(columns='Obama', ascending=False)
droped = obama_bush.dropna()
print droped.head(10)
```

                      Obama  Schiliro
    word                             
    obama         43.295653  9.621256
    law           14.722936  7.361468
    democratic    12.410689  6.205344
    senate        10.164288  3.388096
    presidential   7.386955  3.693478
    president      7.226869  9.033587
    policy         6.095386  3.047693
    states         5.473201  1.824400
    office         5.248173  2.624086
    2011           5.107041  3.404694
    

    C:\Users\SSQ\AppData\Roaming\Python\Python27\site-packages\ipykernel\__main__.py:6: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=.....)
    

The first 10 words should say: Obama, law, democratic, Senate, presidential, president, policy, states, office, 2011.

**Quiz Question**. Among the words that appear in both Barack Obama and Phil Schiliro, take the 5 that have largest weights in Obama. How many of the articles in the Wikipedia dataset contain all of those 5 words?


```python
common_words = set(['obama', 'law', 'democratic', 'senate', 'presidential'])  # YOUR CODE HERE

def has_top_words(word_count_vector):
    # extract the keys of word_count_vector and convert it to a set
    unique_words = set(word_count_vector.keys())   # YOUR CODE HERE
    # return True if common_words is a subset of unique_words
    # return False otherwise
    #print unique_words
    return common_words.issubset(unique_words)  # YOUR CODE HERE

wiki['has_top_words'] = wiki['word_count'].apply(has_top_words)

# use has_top_words column to answer the quiz question
wiki.head(10) # YOUR CODE HERE
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>URI</th>
      <th>name</th>
      <th>text</th>
      <th>word_count</th>
      <th>has_top_words</th>
      <th>tf_idf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>&lt;http://dbpedia.org/resource/Digby_Morrell&gt;</td>
      <td>Digby Morrell</td>
      <td>digby morrell born 10 october 1979 is a former...</td>
      <td>{u'selection': 1, u'carltons': 1, u'being': 1,...</td>
      <td>False</td>
      <td>{u'selection': 3.83657855309, u'carltons': 7.0...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>&lt;http://dbpedia.org/resource/Alfred_J._Lewy&gt;</td>
      <td>Alfred J. Lewy</td>
      <td>alfred j lewy aka sandy lewy graduated from un...</td>
      <td>{u'precise': 1, u'thomas': 1, u'they': 1, u'di...</td>
      <td>False</td>
      <td>{u'precise': 6.44320060696, u'thomas': 3.32027...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>&lt;http://dbpedia.org/resource/Harpdog_Brown&gt;</td>
      <td>Harpdog Brown</td>
      <td>harpdog brown is a singer and harmonica player...</td>
      <td>{u'just': 1, u'issued': 1, u'mainly': 1, u'nom...</td>
      <td>False</td>
      <td>{u'just': 2.70072996871, u'issued': 4.42971703...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>&lt;http://dbpedia.org/resource/Franz_Rottensteiner&gt;</td>
      <td>Franz Rottensteiner</td>
      <td>franz rottensteiner born in waidmannsfeld lowe...</td>
      <td>{u'all': 1, u'bauforschung': 1, u'just': 1, u'...</td>
      <td>False</td>
      <td>{u'all': 1.64311124349, u'bauforschung': 10.98...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>&lt;http://dbpedia.org/resource/G-Enka&gt;</td>
      <td>G-Enka</td>
      <td>henry krvits born 30 december 1974 in tallinn ...</td>
      <td>{u'legendary': 1, u'gangstergenka': 1, u'legen...</td>
      <td>False</td>
      <td>{u'legendary': 4.28085629437, u'gangstergenka'...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>&lt;http://dbpedia.org/resource/Sam_Henderson&gt;</td>
      <td>Sam Henderson</td>
      <td>sam henderson born october 18 1969 is an ameri...</td>
      <td>{u'now': 1, u'since': 1, u'less': 1, u'being':...</td>
      <td>False</td>
      <td>{u'now': 1.96695239252, u'since': 1.4553767173...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>&lt;http://dbpedia.org/resource/Aaron_LaCrate&gt;</td>
      <td>Aaron LaCrate</td>
      <td>aaron lacrate is an american music producer re...</td>
      <td>{u'exclusive': 2, u'played': 1, u'producer': 1...</td>
      <td>False</td>
      <td>{u'exclusive': 10.4551872307, u'played': 1.545...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>&lt;http://dbpedia.org/resource/Trevor_Ferguson&gt;</td>
      <td>Trevor Ferguson</td>
      <td>trevor ferguson aka john farrow born 11 novemb...</td>
      <td>{u'taxi': 1, u'salon': 1, u'gangs': 1, u'being...</td>
      <td>False</td>
      <td>{u'taxi': 6.05202145609, u'salon': 6.134465125...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>&lt;http://dbpedia.org/resource/Grant_Nelson&gt;</td>
      <td>Grant Nelson</td>
      <td>grant nelson born 27 april 1971 in london also...</td>
      <td>{u'houston': 1, u'frankie': 1, u'labels': 1, u...</td>
      <td>False</td>
      <td>{u'houston': 3.93550594216, u'frankie': 6.0377...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>&lt;http://dbpedia.org/resource/Cathy_Caruth&gt;</td>
      <td>Cathy Caruth</td>
      <td>cathy caruth born 1955 is frank h t rhodes pro...</td>
      <td>{u'phenomenon': 1, u'deborash': 1, u'innovativ...</td>
      <td>False</td>
      <td>{u'phenomenon': 5.7500534264, u'deborash': 10....</td>
    </tr>
  </tbody>
</table>
</div>




```python
print len(wiki[wiki['has_top_words']==True])
```

    14
    

Notice the huge difference in this calculation using TF-IDF scores instead  of raw word counts. We've eliminated noise arising from extremely common words.

## Choosing metrics

You may wonder why Joe Biden, Obama's running mate in two presidential elections, is missing from the query results of `model_tf_idf`. Let's find out why. First, compute the distance between TF-IDF features of Obama and Biden.

**Quiz Question**. Compute the Euclidean distance between TF-IDF features of Obama and Biden. Hint: When using Boolean filter in SFrame/SArray, take the index 0 to access the first match.


```python
print euclidean_distances(tf_idf[wiki.index[wiki['name'] == 'Barack Obama'].tolist()], tf_idf[wiki.index[wiki['name'] == 'Joe Biden'].tolist()])
```

    [[ 123.29745601]]
    

The distance is larger than the distances we found for the 10 nearest neighbors, which we repeat here for readability:


```python
#model_tf_idf.query(wiki[wiki['name'] == 'Barack Obama'], label='name', k=10)
```

But one may wonder, is Biden's article that different from Obama's, more so than, say, Schiliro's? It turns out that, when we compute nearest neighbors using the Euclidean distances, we unwittingly favor short articles over long ones. Let us compute the length of each Wikipedia document, and examine the document lengths for the 100 nearest neighbors to Obama's page.


```python
# Comptue length of all documents
def compute_length(row):
    #print row['text'].split(' ')
    return len(row['text'].split(' '))
wiki['length'] = wiki.apply(compute_length, axis=1)

```


```python
wiki.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>URI</th>
      <th>name</th>
      <th>text</th>
      <th>word_count</th>
      <th>has_top_words</th>
      <th>tf_idf</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>&lt;http://dbpedia.org/resource/Digby_Morrell&gt;</td>
      <td>Digby Morrell</td>
      <td>digby morrell born 10 october 1979 is a former...</td>
      <td>{u'selection': 1, u'carltons': 1, u'being': 1,...</td>
      <td>False</td>
      <td>{u'selection': 3.83657855309, u'carltons': 7.0...</td>
      <td>251</td>
    </tr>
    <tr>
      <th>1</th>
      <td>&lt;http://dbpedia.org/resource/Alfred_J._Lewy&gt;</td>
      <td>Alfred J. Lewy</td>
      <td>alfred j lewy aka sandy lewy graduated from un...</td>
      <td>{u'precise': 1, u'thomas': 1, u'they': 1, u'di...</td>
      <td>False</td>
      <td>{u'precise': 6.44320060696, u'thomas': 3.32027...</td>
      <td>223</td>
    </tr>
    <tr>
      <th>2</th>
      <td>&lt;http://dbpedia.org/resource/Harpdog_Brown&gt;</td>
      <td>Harpdog Brown</td>
      <td>harpdog brown is a singer and harmonica player...</td>
      <td>{u'just': 1, u'issued': 1, u'mainly': 1, u'nom...</td>
      <td>False</td>
      <td>{u'just': 2.70072996871, u'issued': 4.42971703...</td>
      <td>226</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Compute 100 nearest neighbors and display their lengths
distances, indices = model_tf_idf.kneighbors(tf_idf[35817], n_neighbors=100)
neighbors = pd.DataFrame(data={'distance':distances.flatten()}, index=indices.flatten())
#print neighbors.head(2)
nearest_neighbors_euclidean = wiki.join(neighbors).sort('distance')[['name', 'length', 'distance']]
print nearest_neighbors_euclidean.head(10)
```

                              name  length    distance
    35817             Barack Obama     540    0.000000
    7914             Phil Schiliro     208  106.861014
    46811            Jeff Sessions     230  108.871674
    44681   Jesse Lee (politician)     216  109.045698
    38376           Samantha Power     310  109.108106
    6507              Bob Menendez     220  109.781867
    38714  Eric Stern (politician)     255  109.957788
    44825           James A. Guest     215  110.413889
    44368     Roland Grossenbacher     201  110.470609
    33417            Tulsi Gabbard     228  110.696998
    

    C:\Users\SSQ\AppData\Roaming\Python\Python27\site-packages\ipykernel\__main__.py:5: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=.....)
    

To see how these document lengths compare to the lengths of other documents in the corpus, let's make a histogram of the document lengths of Obama's 100 nearest neighbors and compare to a histogram of document lengths for all documents.


```python
print wiki['length'][wiki.index[wiki['name'] == 'Barack Obama']].tolist()
```

    [540L]
    


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
      <th>word_count</th>
      <th>has_top_words</th>
      <th>tf_idf</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>&lt;http://dbpedia.org/resource/Digby_Morrell&gt;</td>
      <td>Digby Morrell</td>
      <td>digby morrell born 10 october 1979 is a former...</td>
      <td>{u'selection': 1, u'carltons': 1, u'being': 1,...</td>
      <td>False</td>
      <td>{u'selection': 3.83657855309, u'carltons': 7.0...</td>
      <td>251</td>
    </tr>
    <tr>
      <th>1</th>
      <td>&lt;http://dbpedia.org/resource/Alfred_J._Lewy&gt;</td>
      <td>Alfred J. Lewy</td>
      <td>alfred j lewy aka sandy lewy graduated from un...</td>
      <td>{u'precise': 1, u'thomas': 1, u'they': 1, u'di...</td>
      <td>False</td>
      <td>{u'precise': 6.44320060696, u'thomas': 3.32027...</td>
      <td>223</td>
    </tr>
  </tbody>
</table>
</div>




```python
nearest_neighbors_euclidean.head(2)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>length</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35817</th>
      <td>Barack Obama</td>
      <td>540</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7914</th>
      <td>Phil Schiliro</td>
      <td>208</td>
      <td>106.861014</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python
plt.figure(figsize=(10.5,4.5))
plt.hist(wiki['length'], 50, color='k', edgecolor='None', histtype='stepfilled', normed=True,
         label='Entire Wikipedia', zorder=3, alpha=0.8)
plt.hist(nearest_neighbors_euclidean['length'][:100], 50, color='r', edgecolor='None', histtype='stepfilled', normed=True,
         label='100 NNs of Obama (Euclidean)', zorder=10, alpha=0.8)
plt.axvline(x=wiki['length'][wiki.index[wiki['name'] == 'Barack Obama']].tolist()[0], color='k', linestyle='--', linewidth=4,
           label='Length of Barack Obama', zorder=2)
plt.axvline(x=wiki['length'][wiki.index[wiki['name'] == 'Joe Biden']].tolist()[0], color='g', linestyle='--', linewidth=4,
           label='Length of Joe Biden', zorder=1)
plt.axis([0, 1000, 0, 0.04])

plt.legend(loc='best', prop={'size':15})
plt.title('Distribution of document length')
plt.xlabel('# of words')
plt.ylabel('Percentage')
plt.rcParams.update({'font.size':16})
plt.tight_layout()
```


![png](output_94_0.png)


Relative to the rest of Wikipedia, nearest neighbors of Obama are overwhemingly short, most of them being shorter than 300 words. The bias towards short articles is not appropriate in this application as there is really no reason to  favor short articles over long articles (they are all Wikipedia articles, after all). Many of the Wikipedia articles are 300 words or more, and both Obama and Biden are over 300 words long.

**Note**: For the interest of computation time, the dataset given here contains _excerpts_ of the articles rather than full text. For instance, the actual Wikipedia article about Obama is around 25000 words. Do not be surprised by the low numbers shown in the histogram.

**Note:** Both word-count features and TF-IDF are proportional to word frequencies. While TF-IDF penalizes very common words, longer articles tend to have longer TF-IDF vectors simply because they have more words in them.

To remove this bias, we turn to **cosine distances**:
$$
d(\mathbf{x},\mathbf{y}) = 1 - \frac{\mathbf{x}^T\mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}
$$
Cosine distances let us compare word distributions of two articles of varying lengths.

Let us train a new nearest neighbor model, this time with cosine distances.  We then repeat the search for Obama's 100 nearest neighbors.


```python
model2_tf_idf = NearestNeighbors(algorithm='brute', metric='cosine')
model2_tf_idf.fit(tf_idf)
distances, indices = model2_tf_idf.kneighbors(tf_idf[35817], n_neighbors=100)
neighbors = pd.DataFrame(data={'distance':distances.flatten()}, index=indices.flatten())
nearest_neighbors_cosine = wiki.join(neighbors)[['name', 'length', 'distance']].sort('distance')
print nearest_neighbors_cosine.head(10)
```

                              name  length      distance
    35817             Barack Obama     540 -1.110223e-15
    24478                Joe Biden     414  7.031387e-01
    38376           Samantha Power     310  7.429819e-01
    57108   Hillary Rodham Clinton     580  7.583584e-01
    38714  Eric Stern (politician)     255  7.705612e-01
    46140             Robert Gibbs     257  7.846775e-01
    6796               Eric Holder     232  7.880391e-01
    44681   Jesse Lee (politician)     216  7.909264e-01
    18827             Henry Waxman     279  7.983226e-01
    2412           Joe the Plumber     217  7.994664e-01
    

    C:\Users\SSQ\AppData\Roaming\Python\Python27\site-packages\ipykernel\__main__.py:5: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=.....)
    

From a glance at the above table, things look better.  For example, we now see Joe Biden as Barack Obama's nearest neighbor!  We also see Hillary Clinton on the list.  This list looks even more plausible as nearest neighbors of Barack Obama.

Let's make a plot to better visualize the effect of having used cosine distance in place of Euclidean on our TF-IDF vectors.


```python
plt.figure(figsize=(10.5,4.5))
plt.figure(figsize=(10.5,4.5))
plt.hist(wiki['length'], 50, color='k', edgecolor='None', histtype='stepfilled', normed=True,
         label='Entire Wikipedia', zorder=3, alpha=0.8)
plt.hist(nearest_neighbors_euclidean['length'][:100], 50, color='r', edgecolor='None', histtype='stepfilled', normed=True,
         label='100 NNs of Obama (Euclidean)', zorder=10, alpha=0.8)
plt.hist(nearest_neighbors_cosine['length'][:100], 50, color='b', edgecolor='None', histtype='stepfilled', normed=True,
         label='100 NNs of Obama (cosine)', zorder=11, alpha=0.8)
plt.axvline(x=wiki['length'][wiki.index[wiki['name'] == 'Barack Obama']].tolist()[0], color='k', linestyle='--', linewidth=4,
           label='Length of Barack Obama', zorder=2)
plt.axvline(x=wiki['length'][wiki.index[wiki['name'] == 'Joe Biden']].tolist()[0], color='g', linestyle='--', linewidth=4,
           label='Length of Joe Biden', zorder=1)
plt.axis([0, 1000, 0, 0.04])
plt.legend(loc='best', prop={'size':15})
plt.title('Distribution of document length')
plt.xlabel('# of words')
plt.ylabel('Percentage')
plt.rcParams.update({'font.size': 16})
plt.tight_layout()

```


    <matplotlib.figure.Figure at 0xc2ebb4e0>



![png](output_100_1.png)


Indeed, the 100 nearest neighbors using cosine distance provide a sampling across the range of document lengths, rather than just short articles like Euclidean distance provided.

**Moral of the story**: In deciding the features and distance measures, check if they produce results that make sense for your particular application.

# Problem with cosine distances: tweets vs. long articles

Happily ever after? Not so fast. Cosine distances ignore all document lengths, which may be great in certain situations but not in others. For instance, consider the following (admittedly contrived) example.

```
+--------------------------------------------------------+
|                                             +--------+ |
|  One that shall not be named                | Follow | |
|  @username                                  +--------+ |
|                                                        |
|  Democratic governments control law in response to     |
|  popular act.                                          |
|                                                        |
|  8:05 AM - 16 May 2016                                 |
|                                                        |
|  Reply   Retweet (1,332)   Like (300)                  |
|                                                        |
+--------------------------------------------------------+
```

How similar is this tweet to Barack Obama's Wikipedia article? Let's transform the tweet into TF-IDF features, using an encoder fit to the Wikipedia dataset.  (That is, let's treat this tweet as an article in our Wikipedia dataset and see what happens.)


```python
tweet = {'act': 3.4597778278724887,
 'control': 3.721765211295327,
 'democratic': 3.1026721743330414,
 'governments': 4.167571323949673,
 'in': 0.0009654063501214492,
 'law': 2.4538226269605703,
 'popular': 2.764478952022998,
 'response': 4.261461747058352,
 'to': 0.04694493768179923}
```


```python
map_index_to_word['act']
```




    547084




```python
word_indices = []
for word in tweet.keys():
    if word in map_index_to_word.keys():
        word_indices.append(map_index_to_word[word]  )

tweet_tf_idf = csr_matrix( (list(tweet.values()), ([0]*len(word_indices), word_indices)),
                          shape=(1, tf_idf.shape[1]) )
```


```python
print len(word_indices)
```

    9
    

Let's look at the TF-IDF vectors for this tweet and for Barack Obama's Wikipedia entry, just to visually see their differences.

Now, compute the cosine distance between the Barack Obama article and this tweet:


```python
from sklearn.metrics.pairwise import cosine_distances

obama_tf_idf = tf_idf[35817]
print cosine_distances(obama_tf_idf, tweet_tf_idf)
```

    [[ 0.70591838]]
    

Let's compare this distance to the distance between the Barack Obama article and all of its Wikipedia 10 nearest neighbors:


```python
distances, indices = model2_tf_idf.kneighbors(obama_tf_idf, n_neighbors=10)
print distances
```

    [[ -1.11022302e-15   7.03138677e-01   7.42981902e-01   7.58358398e-01
        7.70561228e-01   7.84677505e-01   7.88039073e-01   7.90926415e-01
        7.98322603e-01   7.99466360e-01]]
    

With cosine distances, the tweet is "nearer" to Barack Obama than everyone else, except for Joe Biden!  This probably is not something we want. If someone is reading the Barack Obama Wikipedia page, would you want to recommend they read this tweet? Ignoring article lengths completely resulted in nonsensical results. In practice, it is common to enforce maximum or minimum document lengths. After all, when someone is reading a long article from _The Atlantic_, you wouldn't recommend him/her a tweet.
