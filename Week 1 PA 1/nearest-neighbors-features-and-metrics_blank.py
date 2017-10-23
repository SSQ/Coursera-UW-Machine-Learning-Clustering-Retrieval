
# coding: utf-8

# # Nearest Neighbors

# When exploring a large set of documents -- such as Wikipedia, news articles, StackOverflow, etc. -- it can be useful to get a list of related material. To find relevant documents you typically
# * Decide on a notion of similarity
# * Find the documents that are most similar 
# 
# In the assignment you will
# * Gain intuition for different notions of similarity and practice finding similar documents. 
# * Explore the tradeoffs with representing documents using raw word counts and TF-IDF
# * Explore the behavior of different distance metrics by looking at the Wikipedia pages most similar to President Obama’s page.

# **Note to Amazon EC2 users**: To conserve memory, make sure to stop all the other notebooks before running this notebook.

# ## Import necessary packages

# As usual we need to first import the Python packages that we will need.
# 
# The following code block will check if you have the correct version of GraphLab Create. Any version later than 1.8.5 will do. To upgrade, read [this page](https://turi.com/download/upgrade-graphlab-create.html).

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.sparse import csr_matrix      # sparse matrices
get_ipython().magic(u'matplotlib inline')


# ## Load Wikipedia dataset

# We will be using the same dataset of Wikipedia pages that we used in the Machine Learning Foundations course (Course 1). Each element of the dataset consists of a link to the wikipedia article, the name of the person, and the text of the article (in lowercase).  

# In[2]:

wiki = pd.read_csv('people_wiki.csv')


# In[3]:

wiki.head(2)


# In[4]:

wiki.shape


# ## Extract word count vectors

# As we have seen in Course 1, we can extract word count vectors using a GraphLab utility function.  We add this as a column in `wiki`.

# In[5]:

def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']
    print data[0],indices[0], indptr[0],shape[0]
    return csr_matrix( (data, indices, indptr), shape)


# In[6]:

word_count = load_sparse_csr('people_wiki_word_count.npz')


# In[7]:

word_count


# In[8]:

type(word_count)


# In[9]:

print word_count[0]


# In[10]:

with open('people_wiki_map_index_to_word.json') as people_wiki_map_index_to_word:    
    map_index_to_word = json.load(people_wiki_map_index_to_word)


# In[11]:

len(map_index_to_word)


# In[12]:

wiki.head(2)


# In[ ]:




# ## Find nearest neighbors

# Let's start by finding the nearest neighbors of the Barack Obama page using the word count vectors to represent the articles and Euclidean distance to measure distance.  For this, again will we use a GraphLab Create implementation of nearest neighbor search.

# In[13]:

from sklearn.neighbors import NearestNeighbors

model = NearestNeighbors(metric='euclidean', algorithm='brute')
model.fit(word_count)


# In[15]:

wiki[wiki['name'] == 'Barack Obama']


# In[16]:

distances, indices = model.kneighbors(word_count[35817], n_neighbors=10) # 1st arg: word count vector


# In[28]:

neighbors = pd.DataFrame(data={'distance':distances.flatten()},index=indices.flatten())
print wiki.join(neighbors).sort('distance')[['name','distance']][0:10]


# In[29]:

neighbors.head(2)


# Let's look at the top 10 nearest neighbors by performing the following query:

# In[ ]:

# model.query(wiki[wiki['name']=='Barack Obama'], label='name', k=10)


# All of the 10 people are politicians, but about half of them have rather tenuous connections with Obama, other than the fact that they are politicians.
# 
# * Francisco Barrio is a Mexican politician, and a former governor of Chihuahua.
# * Walter Mondale and Don Bonker are Democrats who made their career in late 1970s.
# * Wynn Normington Hugh-Jones is a former British diplomat and Liberal Party official.
# * Andy Anstett is a former politician in Manitoba, Canada.
# 
# Nearest neighbors with raw word counts got some things right, showing all politicians in the query result, but missed finer and important details.
# 
# For instance, let's find out why Francisco Barrio was considered a close neighbor of Obama.  To do this, let's look at the most frequently used words in each of Barack Obama and Francisco Barrio's pages:

# In[31]:

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


# In[32]:

wiki.head(3)


# In[67]:

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


# In[73]:

name = 'Barack Obama'
row = wiki[wiki['name'] == name]
dic = row['word_count'].iloc[0]
#print dic
word_count_ = pd.DataFrame(dic.items(), columns=['word','count'], index=None)
print word_count_.head(4)


# In[176]:

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


# Let's extract the list of most frequent words that appear in both Obama's and Barrio's documents. We've so far sorted all words from Obama and Barrio's articles by their word frequencies. We will now use a dataframe operation known as **join**. The **join** operation is very useful when it comes to playing around with data: it lets you combine the content of two tables using a shared column (in this case, the word column). See [the documentation](https://dato.com/products/create/docs/generated/graphlab.SFrame.join.html) for more details.
# 
# For instance, running
# ```
# obama_words.join(barrio_words, on='word')
# ```
# will extract the rows from both tables that correspond to the common words.

# In[177]:

combined_words = obama_words.set_index('word').join(barrio_words.set_index('word'), lsuffix='_obama', rsuffix='_barrio')
combined_words.head(10)


# In[101]:

"""
combined_words = obama_words.join(barrio_words, how='inner', on='word')
combined_words
"""


# Since both tables contained the column named `count`, SFrame automatically renamed one of them to prevent confusion. Let's rename the columns to tell which one is for which. By inspection, we see that the first column (`count`) is for Obama and the second (`count.1`) for Barrio.

# In[106]:

combined_words = combined_words.rename(index=str, columns={'count_obama':'Obama', 'count_barrio':'Barrio'})
combined_words.head(10)


# In[ ]:




# **Note**. The **join** operation does not enforce any particular ordering on the shared column. So to obtain, say, the five common words that appear most often in Obama's article, sort the combined table by the Obama column. Don't forget `ascending=False` to display largest counts first.

# In[109]:

combined_words.sort(columns='Obama', ascending=False)
combined_words.head(10)


# **Quiz Question**. Among the words that appear in both Barack Obama and Francisco Barrio, take the 5 that appear most frequently in Obama. How many of the articles in the Wikipedia dataset contain all of those 5 words?
# 
# Hint:
# * Refer to the previous paragraph for finding the words that appear in both articles. Sort the common words by their frequencies in Obama's article and take the largest five.
# * Each word count vector is a Python dictionary. For each word count vector in SFrame, you'd have to check if the set of the 5 common words is a subset of the keys of the word count vector. Complete the function `has_top_words` to accomplish the task.
#   - Convert the list of top 5 words into set using the syntax
# ```
# set(common_words)
# ```
#     where `common_words` is a Python list. See [this link](https://docs.python.org/2/library/stdtypes.html#set) if you're curious about Python sets.
#   - Extract the list of keys of the word count dictionary by calling the [`keys()` method](https://docs.python.org/2/library/stdtypes.html#dict.keys).
#   - Convert the list of keys into a set as well.
#   - Use [`issubset()` method](https://docs.python.org/2/library/stdtypes.html#set) to check if all 5 words are among the keys.
# * Now apply the `has_top_words` function on every row of the SFrame.
# * Compute the sum of the result column to obtain the number of articles containing all the 5 top words.

# In[135]:

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


# In[178]:

len(wiki[wiki['has_top_words'] == True])


# **Checkpoint**. Check your `has_top_words` function on two random articles:

# In[127]:

set(['the']).issubset(set(wiki.iloc[32]['word_count'].keys()))


# In[134]:

wiki.iloc[32]['word_count']['to']


# In[138]:

common_words = set(['the', 'in', 'and', 'of', 'to'])
for i in common_words:
    print i, set([i]).issubset(set(wiki.iloc[32]['word_count'].keys()))


# In[139]:

print 'Output from your function:', has_top_words(wiki.iloc[32]['word_count'])
print 'Correct output: True'
print 'Also check the length of unique_words. It should be 167'


# In[140]:

print 'Output from your function:', has_top_words(wiki.iloc[33]['word_count'])
print 'Correct output: False'
print 'Also check the length of unique_words. It should be 188'


# **Quiz Question**. Measure the pairwise distance between the Wikipedia pages of Barack Obama, George W. Bush, and Joe Biden. Which of the three pairs has the smallest distance?
# 
# Hint: For this question, take the row vectors from the word count matrix that correspond to Obama, Bush, and Biden. To compute the Euclidean distance between any two sparse vectors, use [sklearn.metrics.pairwise.euclidean_distances](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html).

# In[145]:

print wiki.index[wiki['name'] == 'Barack Obama'].tolist()


# In[146]:

print wiki.index[wiki['name'] == 'George W. Bush'].tolist()


# In[147]:

print wiki.index[wiki['name'] == 'Joe Biden'].tolist()


# In[151]:

from sklearn.metrics.pairwise import euclidean_distances
print euclidean_distances(word_count[wiki.index[wiki['name'] == 'Barack Obama'].tolist()], word_count[wiki.index[wiki['name'] == 'George W. Bush'].tolist()])
print euclidean_distances(word_count[wiki.index[wiki['name'] == 'Barack Obama'].tolist()], word_count[wiki.index[wiki['name'] == 'Joe Biden'].tolist()])
print euclidean_distances(word_count[wiki.index[wiki['name'] == 'George W. Bush'].tolist()], word_count[wiki.index[wiki['name'] == 'Joe Biden'].tolist()])


# **Quiz Question**. Collect all words that appear both in Barack Obama and George W. Bush pages.  Out of those words, find the 10 words that show up most often in Obama's page. 

# In[170]:

obama_words = top_words('Barack Obama')
bush_words = top_words('George W. Bush')

obama_bush = obama_words.set_index('word').join(bush_words.set_index('word'), lsuffix='_obama', rsuffix='_bush')
obama_bush = obama_bush.rename(index=str, columns={'count_obama':'Obama', 'count_bush':'Bush'})
obama_bush.sort(columns='Obama', ascending=False)
dropped = obama_bush.dropna()
print dropped.head(10)


# In[ ]:




# **Note.** Even though common words are swamping out important subtle differences, commonalities in rarer political words still matter on the margin. This is why politicians are being listed in the query result instead of musicians, for example. In the next subsection, we will introduce a different metric that will place greater emphasis on those rarer words.

# ## TF-IDF to the rescue

# Much of the perceived commonalities between Obama and Barrio were due to occurrences of extremely frequent words, such as "the", "and", and "his". So nearest neighbors is recommending plausible results sometimes for the wrong reasons. 
# 
# To retrieve articles that are more relevant, we should focus more on rare words that don't happen in every article. **TF-IDF** (term frequency–inverse document frequency) is a feature representation that penalizes words that are too common.  Let's use GraphLab Create's implementation of TF-IDF and repeat the search for the 10 nearest neighbors of Barack Obama:

# In[155]:

#wiki['tf_idf'] = graphlab.text_analytics.tf_idf(wiki['word_count'])
tf_idf = load_sparse_csr('people_wiki_tf_idf.npz')
wiki['tf_idf'] = unpack_dict(tf_idf, map_index_to_word)


# In[156]:

model_tf_idf = NearestNeighbors(metric='euclidean', algorithm='brute')
model_tf_idf.fit(tf_idf)


# In[157]:

distances, indices = model_tf_idf.kneighbors(tf_idf[35817], n_neighbors=10)


# In[159]:

neighbors = pd.DataFrame(data={'distance':distances.flatten()}, index=indices.flatten())

print wiki.join(neighbors).sort('distance')[['name','distance']][0:10]


# In[ ]:

#model_tf_idf.query(wiki[wiki['name'] == 'Barack Obama'], label='name', k=10)


# Let's determine whether this list makes sense.
# * With a notable exception of Roland Grossenbacher, the other 8 are all American politicians who are contemporaries of Barack Obama.
# * Phil Schiliro, Jesse Lee, Samantha Power, and Eric Stern worked for Obama.
# 
# Clearly, the results are more plausible with the use of TF-IDF. Let's take a look at the word vector for Obama and Schilirio's pages. Notice that TF-IDF representation assigns a weight to each word. This weight captures relative importance of that word in the document. Let us sort the words in Obama's article by their TF-IDF weights; we do the same for Schiliro's article as well.

# In[160]:

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


# Using the **join** operation we learned earlier, try your hands at computing the common words shared by Obama's and Schiliro's articles. Sort the common words by their TF-IDF weights in Obama's document.

# In[169]:

obama_words = top_words_tf_idf('Barack Obama')
schiliro_words = top_words_tf_idf('Phil Schiliro')

obama_bush = obama_words.set_index('word').join(schiliro_words.set_index('word'), lsuffix='_obama', rsuffix='_schiliro')
obama_bush = obama_bush.rename(index=str, columns={'weight_obama':'Obama', 'weight_schiliro':'Schiliro'})
obama_bush.sort(columns='Obama', ascending=False)
droped = obama_bush.dropna()
print droped.head(10)


# The first 10 words should say: Obama, law, democratic, Senate, presidential, president, policy, states, office, 2011.

# **Quiz Question**. Among the words that appear in both Barack Obama and Phil Schiliro, take the 5 that have largest weights in Obama. How many of the articles in the Wikipedia dataset contain all of those 5 words?

# In[179]:

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


# In[180]:

print len(wiki[wiki['has_top_words']==True])


# Notice the huge difference in this calculation using TF-IDF scores instead  of raw word counts. We've eliminated noise arising from extremely common words.

# ## Choosing metrics

# You may wonder why Joe Biden, Obama's running mate in two presidential elections, is missing from the query results of `model_tf_idf`. Let's find out why. First, compute the distance between TF-IDF features of Obama and Biden.

# **Quiz Question**. Compute the Euclidean distance between TF-IDF features of Obama and Biden. Hint: When using Boolean filter in SFrame/SArray, take the index 0 to access the first match.

# In[182]:

print euclidean_distances(tf_idf[wiki.index[wiki['name'] == 'Barack Obama'].tolist()], tf_idf[wiki.index[wiki['name'] == 'Joe Biden'].tolist()])


# The distance is larger than the distances we found for the 10 nearest neighbors, which we repeat here for readability:

# In[ ]:

#model_tf_idf.query(wiki[wiki['name'] == 'Barack Obama'], label='name', k=10)


# But one may wonder, is Biden's article that different from Obama's, more so than, say, Schiliro's? It turns out that, when we compute nearest neighbors using the Euclidean distances, we unwittingly favor short articles over long ones. Let us compute the length of each Wikipedia document, and examine the document lengths for the 100 nearest neighbors to Obama's page.

# In[190]:

# Comptue length of all documents
def compute_length(row):
    #print row['text'].split(' ')
    return len(row['text'].split(' '))
wiki['length'] = wiki.apply(compute_length, axis=1)


# In[191]:

wiki.head(3)


# In[195]:

# Compute 100 nearest neighbors and display their lengths
distances, indices = model_tf_idf.kneighbors(tf_idf[35817], n_neighbors=100)
neighbors = pd.DataFrame(data={'distance':distances.flatten()}, index=indices.flatten())
#print neighbors.head(2)
nearest_neighbors_euclidean = wiki.join(neighbors).sort('distance')[['name', 'length', 'distance']]
print nearest_neighbors_euclidean.head(10)


# To see how these document lengths compare to the lengths of other documents in the corpus, let's make a histogram of the document lengths of Obama's 100 nearest neighbors and compare to a histogram of document lengths for all documents.

# In[201]:

print wiki['length'][wiki.index[wiki['name'] == 'Barack Obama']].tolist()


# In[203]:

wiki.head(2)


# In[207]:

nearest_neighbors_euclidean.head(2)


# In[ ]:




# In[216]:

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


# Relative to the rest of Wikipedia, nearest neighbors of Obama are overwhemingly short, most of them being shorter than 300 words. The bias towards short articles is not appropriate in this application as there is really no reason to  favor short articles over long articles (they are all Wikipedia articles, after all). Many of the Wikipedia articles are 300 words or more, and both Obama and Biden are over 300 words long.
# 
# **Note**: For the interest of computation time, the dataset given here contains _excerpts_ of the articles rather than full text. For instance, the actual Wikipedia article about Obama is around 25000 words. Do not be surprised by the low numbers shown in the histogram.

# **Note:** Both word-count features and TF-IDF are proportional to word frequencies. While TF-IDF penalizes very common words, longer articles tend to have longer TF-IDF vectors simply because they have more words in them.

# To remove this bias, we turn to **cosine distances**:
# $$
# d(\mathbf{x},\mathbf{y}) = 1 - \frac{\mathbf{x}^T\mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}
# $$
# Cosine distances let us compare word distributions of two articles of varying lengths.
# 
# Let us train a new nearest neighbor model, this time with cosine distances.  We then repeat the search for Obama's 100 nearest neighbors.

# In[218]:

model2_tf_idf = NearestNeighbors(algorithm='brute', metric='cosine')
model2_tf_idf.fit(tf_idf)
distances, indices = model2_tf_idf.kneighbors(tf_idf[35817], n_neighbors=100)
neighbors = pd.DataFrame(data={'distance':distances.flatten()}, index=indices.flatten())
nearest_neighbors_cosine = wiki.join(neighbors)[['name', 'length', 'distance']].sort('distance')
print nearest_neighbors_cosine.head(10)


# From a glance at the above table, things look better.  For example, we now see Joe Biden as Barack Obama's nearest neighbor!  We also see Hillary Clinton on the list.  This list looks even more plausible as nearest neighbors of Barack Obama.
# 
# Let's make a plot to better visualize the effect of having used cosine distance in place of Euclidean on our TF-IDF vectors.

# In[221]:

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


# Indeed, the 100 nearest neighbors using cosine distance provide a sampling across the range of document lengths, rather than just short articles like Euclidean distance provided.

# **Moral of the story**: In deciding the features and distance measures, check if they produce results that make sense for your particular application.

# # Problem with cosine distances: tweets vs. long articles

# Happily ever after? Not so fast. Cosine distances ignore all document lengths, which may be great in certain situations but not in others. For instance, consider the following (admittedly contrived) example.

# ```
# +--------------------------------------------------------+
# |                                             +--------+ |
# |  One that shall not be named                | Follow | |
# |  @username                                  +--------+ |
# |                                                        |
# |  Democratic governments control law in response to     |
# |  popular act.                                          |
# |                                                        |
# |  8:05 AM - 16 May 2016                                 |
# |                                                        |
# |  Reply   Retweet (1,332)   Like (300)                  |
# |                                                        |
# +--------------------------------------------------------+
# ```

# How similar is this tweet to Barack Obama's Wikipedia article? Let's transform the tweet into TF-IDF features, using an encoder fit to the Wikipedia dataset.  (That is, let's treat this tweet as an article in our Wikipedia dataset and see what happens.)

# In[222]:

tweet = {'act': 3.4597778278724887,
 'control': 3.721765211295327,
 'democratic': 3.1026721743330414,
 'governments': 4.167571323949673,
 'in': 0.0009654063501214492,
 'law': 2.4538226269605703,
 'popular': 2.764478952022998,
 'response': 4.261461747058352,
 'to': 0.04694493768179923}


# In[228]:

map_index_to_word['act']


# In[230]:

word_indices = []
for word in tweet.keys():
    if word in map_index_to_word.keys():
        word_indices.append(map_index_to_word[word]  )

tweet_tf_idf = csr_matrix( (list(tweet.values()), ([0]*len(word_indices), word_indices)),
                          shape=(1, tf_idf.shape[1]) )


# In[231]:

print len(word_indices)


# Let's look at the TF-IDF vectors for this tweet and for Barack Obama's Wikipedia entry, just to visually see their differences.

# Now, compute the cosine distance between the Barack Obama article and this tweet:

# In[232]:

from sklearn.metrics.pairwise import cosine_distances

obama_tf_idf = tf_idf[35817]
print cosine_distances(obama_tf_idf, tweet_tf_idf)


# Let's compare this distance to the distance between the Barack Obama article and all of its Wikipedia 10 nearest neighbors:

# In[233]:

distances, indices = model2_tf_idf.kneighbors(obama_tf_idf, n_neighbors=10)
print distances


# With cosine distances, the tweet is "nearer" to Barack Obama than everyone else, except for Joe Biden!  This probably is not something we want. If someone is reading the Barack Obama Wikipedia page, would you want to recommend they read this tweet? Ignoring article lengths completely resulted in nonsensical results. In practice, it is common to enforce maximum or minimum document lengths. After all, when someone is reading a long article from _The Atlantic_, you wouldn't recommend him/her a tweet.
