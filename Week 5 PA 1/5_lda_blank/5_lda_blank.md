
# Latent Dirichlet Allocation for Text Data

In this assignment you will

* apply standard preprocessing techniques on Wikipedia text data
* use GraphLab Create to fit a Latent Dirichlet allocation (LDA) model
* explore and interpret the results, including topic keywords and topic assignments for documents

Recall that a major feature distinguishing the LDA model from our previously explored methods is the notion of *mixed membership*. Throughout the course so far, our models have assumed that each data point belongs to a single cluster. k-means determines membership simply by shortest distance to the cluster center, and Gaussian mixture models suppose that each data point is drawn from one of their component mixture distributions. In many cases, though, it is more realistic to think of data as genuinely belonging to more than one cluster or category - for example, if we have a model for text data that includes both "Politics" and "World News" categories, then an article about a recent meeting of the United Nations should have membership in both categories rather than being forced into just one.

With this in mind, we will use GraphLab Create tools to fit an LDA model to a corpus of Wikipedia articles and examine the results to analyze the impact of a mixed membership approach. In particular, we want to identify the topics discovered by the model in terms of their most important words, and we want to use the model to predict the topic membership distribution for a given document. 

**Note to Amazon EC2 users**: To conserve memory, make sure to stop all the other notebooks before running this notebook.

## Text Data Preprocessing
We'll start by importing our familiar Wikipedia dataset.

The following code block will check if you have the correct version of GraphLab Create. Any version later than 1.8.5 will do. To upgrade, read [this page](https://turi.com/download/upgrade-graphlab-create.html).


```python
import graphlab as gl
import numpy as np
import matplotlib.pyplot as plt 

%matplotlib inline

'''Check GraphLab Create version'''
from distutils.version import StrictVersion
assert (StrictVersion(gl.version) >= StrictVersion('1.8.5')), 'GraphLab Create must be version 1.8.5 or later.'
```


```python
# import wiki data
wiki = gl.SFrame('people_wiki.gl/')
wiki
```

    This non-commercial license of GraphLab Create for academic use is assigned to ssq6554@gmail.com and will expire on November 12, 2018.
    

    [INFO] graphlab.cython.cy_server: GraphLab Create v2.1 started. Logging: C:\Users\Beginner\AppData\Local\Temp\graphlab_server_1510404145.log.0
    




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



In the original data, each Wikipedia article is represented by a URI, a name, and a string containing the entire text of the article. Recall from the video lectures that LDA requires documents to be represented as a _bag of words_, which ignores word ordering in the document but retains information on how many times each word appears. As we have seen in our previous encounters with text data, words such as 'the', 'a', or 'and' are by far the most frequent, but they appear so commonly in the English language that they tell us almost nothing about how similar or dissimilar two documents might be. 

Therefore, before we train our LDA model, we will preprocess the Wikipedia data in two steps: first, we will create a bag of words representation for each article, and then we will remove the common words that don't help us to distinguish between documents. For both of these tasks we can use pre-implemented tools from GraphLab Create:


```python
wiki_docs = gl.text_analytics.count_words(wiki['text'])
wiki_docs = wiki_docs.dict_trim_by_keys(gl.text_analytics.stopwords(), exclude=True)
```

## Model fitting and interpretation
In the video lectures we saw that Gibbs sampling can be used to perform inference in the LDA model. In this assignment we will use a GraphLab Create method to learn the topic model for our Wikipedia data, and our main emphasis will be on interpreting the results. We'll begin by creating the topic model using create() from GraphLab Create's topic_model module.

Note: This may take several minutes to run.


```python
topic_model = gl.topic_model.create(wiki_docs, num_topics=10, num_iterations=200)
```


<pre>Learning a topic model</pre>



<pre>       Number of documents     59071</pre>



<pre>           Vocabulary size    547462</pre>



<pre>   Running collapsed Gibbs sampling</pre>



<pre>+-----------+---------------+----------------+-----------------+</pre>



<pre>| Iteration | Elapsed Time  | Tokens/Second  | Est. Perplexity |</pre>



<pre>+-----------+---------------+----------------+-----------------+</pre>



<pre>| 10        | 13.87s        | 6.5182e+006    | 0               |</pre>



<pre>| 20        | 26.79s        | 6.91004e+006   | 0               |</pre>



<pre>| 30        | 39.58s        | 6.20024e+006   | 0               |</pre>



<pre>| 40        | 51.85s        | 6.12554e+006   | 0               |</pre>



<pre>| 50        | 1m 4s         | 6.35525e+006   | 0               |</pre>



<pre>| 60        | 1m 17s        | 5.95443e+006   | 0               |</pre>



<pre>| 70        | 1m 30s        | 5.534e+006     | 0               |</pre>



<pre>| 80        | 1m 43s        | 6.251e+006     | 0               |</pre>



<pre>| 90        | 1m 55s        | 6.17606e+006   | 0               |</pre>



<pre>| 100       | 2m 7s         | 6.51601e+006   | 0               |</pre>



<pre>| 110       | 2m 20s        | 6.27274e+006   | 0               |</pre>



<pre>| 120       | 2m 32s        | 7.07268e+006   | 0               |</pre>



<pre>| 130       | 2m 45s        | 5.9831e+006    | 0               |</pre>



<pre>| 140       | 2m 57s        | 6.37668e+006   | 0               |</pre>



<pre>| 150       | 3m 10s        | 6.31067e+006   | 0               |</pre>



<pre>| 160       | 3m 22s        | 6.10101e+006   | 0               |</pre>



<pre>| 170       | 3m 34s        | 6.68973e+006   | 0               |</pre>



<pre>| 180       | 3m 47s        | 6.43569e+006   | 0               |</pre>



<pre>| 190       | 3m 59s        | 6.20024e+006   | 0               |</pre>



<pre>| 200       | 4m 11s        | 6.77893e+006   | 0               |</pre>



<pre>+-----------+---------------+----------------+-----------------+</pre>


GraphLab provides a useful summary of the model we have fitted, including the hyperparameter settings for alpha, gamma (note that GraphLab Create calls this parameter beta), and K (the number of topics); the structure of the output data; and some useful methods for understanding the results.


```python
topic_model
```




    Class                          : TopicModel
    
    Schema
    ------
    Vocabulary Size                : 547462
    
    Settings
    --------
    Number of Topics               : 10
    alpha                          : 5.0
    beta                           : 0.1
    Iterations                     : 200
    Training time                  : 252.8128
    Verbose                        : False
    
    Accessible fields             : 
    m['topics']                   : An SFrame containing the topics.
    m['vocabulary']               : An SArray containing the words in the vocabulary.
    Useful methods                : 
    m.get_topics()                : Get the most probable words per topic.
    m.predict(new_docs)           : Make predictions for new documents.



It is certainly useful to have pre-implemented methods available for LDA, but as with our previous methods for clustering and retrieval, implementing and fitting the model gets us only halfway towards our objective. We now need to analyze the fitted model to understand what it has done with our data and whether it will be useful as a document classification system. This can be a challenging task in itself, particularly when the model that we use is complex. We will begin by outlining a sequence of objectives that will help us understand our model in detail. In particular, we will

* get the top words in each topic and use these to identify topic themes
* predict topic distributions for some example documents
* compare the quality of LDA "nearest neighbors" to the NN output from the first assignment
* understand the role of model hyperparameters alpha and gamma

## Load a fitted topic model
The method used to fit the LDA model is a _randomized algorithm_, which means that it involves steps that are random; in this case, the randomness comes from Gibbs sampling, as discussed in the LDA video lectures. Because of these random steps, the algorithm will be expected to yield slighty different output for different runs on the same data - note that this is different from previously seen algorithms such as k-means or EM, which will always produce the same results given the same input and initialization.

It is important to understand that variation in the results is a fundamental feature of randomized methods. However, in the context of this assignment this variation makes it difficult to evaluate the correctness of your analysis, so we will load and analyze a pre-trained model. 

We recommend that you spend some time exploring your own fitted topic model and compare our analysis of the pre-trained model to the same analysis applied to the model you trained above.


```python
topic_model = gl.load_model('topic_models/lda_assignment_topic_model')
```

# Identifying topic themes by top words

We'll start by trying to identify the topics learned by our model with some major themes. As a preliminary check on the results of applying this method, it is reasonable to hope that the model has been able to learn topics that correspond to recognizable categories. In order to do this, we must first recall what exactly a 'topic' is in the context of LDA. 

In the video lectures on LDA we learned that a topic is a probability distribution over words in the vocabulary; that is, each topic assigns a particular probability to every one of the unique words that appears in our data. Different topics will assign different probabilities to the same word: for instance, a topic that ends up describing science and technology articles might place more probability on the word 'university' than a topic that describes sports or politics. Looking at the highest probability words in each topic will thus give us a sense of its major themes. Ideally we would find that each topic is identifiable with some clear theme _and_ that all the topics are relatively distinct.

We can use the GraphLab Create function get_topics() to view the top words (along with their associated probabilities) from each topic.

__Quiz Question:__ Identify the top 3 most probable words for the first topic. 


```python
topic_model.get_topics()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">topic</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">score</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">university</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0337723780773</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">research</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0120334992502</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">professor</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0118011432268</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">international</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.00881383389898</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">institute</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.00851045584573</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">played</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0189511138484</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">season</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0180754216578</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">league</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.017415688373</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">team</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0125104570691</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">career</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0108564659198</td>
    </tr>
</table>
[50 rows x 3 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>




```python
topic_model['topics'] 
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">topic_probabilities</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">vocabulary</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[8.76815182796e-08,<br>1.77848510673e-06, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">melbourne</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.00340476103631,<br>0.00095453836371, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">college</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[8.76815182796e-08,<br>2.88792105425e-05, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">parade</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.0040018721758,<br>8.46897669869e-08, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">education</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.000513024563454,<br>8.46897669869e-08, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">teaches</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[8.76815182796e-08,<br>9.31587436856e-07, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">edflhe</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[8.76815182796e-08,<br>1.61757454945e-05, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">coburg</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[8.76815182796e-08,<br>0.00108157301419, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">coached</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[8.76815182796e-08,<br>0.00063102345382, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">premiership</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.000192110206551,<br>0.00275334901451, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">division</td>
    </tr>
</table>
[547462 rows x 2 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>




```python
topic_model['topics']['topic_probabilities'][0]
```




    array('d', [8.768151827956483e-08, 1.7784851067259479e-06, 7.120614378062247e-08, 9.20897484588228e-08, 7.912267511932995e-08, 8.834451392088384e-08, 9.849798453400219e-08, 1.0167931523455393e-07, 0.0022168099897941326, 1.04737025229335e-07])



__ Quiz Question:__ What is the sum of the probabilities assigned to the top 50 words in the 3rd topic?


```python
topic_model['num_topics']
```




    10L




```python
top_50_words_3rd_topic = topic_model.get_topics([2], num_words=50, output_type='topic_probabilities')
top_50_words_3rd_topic
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">topic</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">score</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">film</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0161482004927</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">music</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0114820618908</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">album</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.00996964339687</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">released</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.00890297536304</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">band</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.00790751347298</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">television</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.00688570530973</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">series</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.00616153882748</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">show</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.00570581950729</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">award</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.00500159074529</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">appeared</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.00492611223289</td>
    </tr>
</table>
[50 rows x 3 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>




```python
top_50_words_3rd_topic['score'].sum()
```




    0.21034366078939654



Let's look at the top 10 words for each topic to see if we can identify any themes:


```python
[x['words'] for x in topic_model.get_topics(output_type='topic_words', num_words=10)]
```


<pre>Less than 4 successfully started. Using only 3 workers.</pre>



<pre>All operations will proceed as normal, but lambda operations will not be able to use all available cores.</pre>



<pre>(The location of the log file is printed at the start of the GraphLab server).</pre>





    [['university',
      'research',
      'professor',
      'international',
      'institute',
      'science',
      'society',
      'studies',
      'director',
      'national'],
     ['played',
      'season',
      'league',
      'team',
      'career',
      'football',
      'games',
      'player',
      'coach',
      'game'],
     ['film',
      'music',
      'album',
      'released',
      'band',
      'television',
      'series',
      'show',
      'award',
      'appeared'],
     ['university',
      'school',
      'served',
      'college',
      'state',
      'american',
      'states',
      'united',
      'born',
      'law'],
     ['member',
      'party',
      'election',
      'minister',
      'government',
      'elected',
      'served',
      'president',
      'general',
      'committee'],
     ['work',
      'art',
      'book',
      'published',
      'york',
      'magazine',
      'radio',
      'books',
      'award',
      'arts'],
     ['company',
      'business',
      'years',
      'group',
      'time',
      'family',
      'people',
      'india',
      'million',
      'indian'],
     ['world',
      'won',
      'born',
      'time',
      'year',
      'team',
      'championship',
      'tour',
      'championships',
      'title'],
     ['born',
      'british',
      'london',
      'australian',
      'south',
      'joined',
      'years',
      'made',
      'england',
      'australia'],
     ['music',
      'de',
      'born',
      'international',
      'la',
      'orchestra',
      'opera',
      'studied',
      'french',
      'festival']]



We propose the following themes for each topic:

- topic 0: Science and research
- topic 1: Team sports
- topic 2: Music, TV, and film
- topic 3: American college and politics
- topic 4: General politics
- topic 5: Art and publishing
- topic 6: Business
- topic 7: International athletics
- topic 8: Great Britain and Australia
- topic 9: International music

We'll save these themes for later:


```python
themes = ['science and research','team sports','music, TV, and film','American college and politics','general politics', \
         'art and publishing','Business','international athletics','Great Britain and Australia','international music']
```

### Measuring the importance of top words

We can learn more about topics by exploring how they place probability mass (which we can think of as a weight) on each of their top words.

We'll do this with two visualizations of the weights for the top words in each topic:
 - the weights of the top 100 words, sorted by the size
 - the total weight of the top 10 words


Here's a plot for the top 100 words by weight in each topic:


```python
for i in range(10):
    plt.plot(range(100), topic_model.get_topics(topic_ids=[i], num_words=100)['score'])
plt.xlabel('Word rank')
plt.ylabel('Probability')
plt.title('Probabilities of Top 100 Words in each Topic')
```




    <matplotlib.text.Text at 0x17292c18>




![png](output_28_1.png)


In the above plot, each line corresponds to one of our ten topics. Notice how for each topic, the weights drop off sharply as we move down the ranked list of most important words. This shows that the top 10-20 words in each topic are assigned a much greater weight than the remaining words - and remember from the summary of our topic model that our vocabulary has 547462 words in total!


Next we plot the total weight assigned by each topic to its top 10 words: 


```python
top_probs = [sum(topic_model.get_topics(topic_ids=[i], num_words=10)['score']) for i in range(10)]

ind = np.arange(10)
width = 0.5

fig, ax = plt.subplots()

ax.bar(ind-(width/2),top_probs,width)
ax.set_xticks(ind)

plt.xlabel('Topic')
plt.ylabel('Probability')
plt.title('Total Probability of Top 10 Words in each Topic')
plt.xlim(-0.5,9.5)
plt.ylim(0,0.15)
plt.show()
```


![png](output_30_0.png)


Here we see that, for our topic model, the top 10 words only account for a small fraction (in this case, between 5% and 13%) of their topic's total probability mass. So while we can use the top words to identify broad themes for each topic, we should keep in mind that in reality these topics are more complex than a simple 10-word summary.

Finally, we observe that some 'junk' words appear highly rated in some topics despite our efforts to remove unhelpful words before fitting the model; for example, the word 'born' appears as a top 10 word in three different topics, but it doesn't help us describe these topics at all.

# Topic distributions for some example documents

As we noted in the introduction to this assignment, LDA allows for mixed membership, which means that each document can partially belong to several different topics. For each document, topic membership is expressed as a vector of weights that sum to one; the magnitude of each weight indicates the degree to which the document represents that particular topic.

We'll explore this in our fitted model by looking at the topic distributions for a few example Wikipedia articles from our data set. We should find that these articles have the highest weights on the topics whose themes are most relevant to the subject of the article - for example, we'd expect an article on a politician to place relatively high weight on topics related to government, while an article about an athlete should place higher weight on topics related to sports or competition.

Topic distributions for documents can be obtained using GraphLab Create's predict() function. GraphLab Create uses a collapsed Gibbs sampler similar to the one described in the video lectures, where only the word assignments variables are sampled.  To get a document-specific topic proportion vector post-facto, predict() draws this vector from the conditional distribution given the sampled word assignments in the document.  Notice that, since these are draws from a _distribution_ over topics that the model has learned, we will get slightly different predictions each time we call this function on a document - we can see this below, where we predict the topic distribution for the article on Barack Obama:


```python
obama = gl.SArray([wiki_docs[int(np.where(wiki['name']=='Barack Obama')[0])]])
pred1 = topic_model.predict(obama, output_type='probability')
pred2 = topic_model.predict(obama, output_type='probability')
print(gl.SFrame({'topics':themes, 'predictions (first draw)':pred1[0], 'predictions (second draw)':pred2[0]}))
```

    +--------------------------+---------------------------+
    | predictions (first draw) | predictions (second draw) |
    +--------------------------+---------------------------+
    |     0.0349462365591      |      0.0322580645161      |
    |     0.0645161290323      |      0.0403225806452      |
    |     0.0215053763441      |      0.0188172043011      |
    |      0.123655913978      |       0.137096774194      |
    |      0.612903225806      |       0.626344086022      |
    |     0.0241935483871      |      0.0161290322581      |
    |     0.0376344086022      |      0.0510752688172      |
    |     0.0188172043011      |      0.0349462365591      |
    |     0.0322580645161      |      0.0241935483871      |
    |     0.0295698924731      |      0.0188172043011      |
    +--------------------------+---------------------------+
    +-------------------------------+
    |             topics            |
    +-------------------------------+
    |      science and research     |
    |          team sports          |
    |      music, TV, and film      |
    | American college and politics |
    |        general politics       |
    |       art and publishing      |
    |            Business           |
    |    international athletics    |
    |  Great Britain and Australia  |
    |      international music      |
    +-------------------------------+
    [10 rows x 3 columns]
    
    

To get a more robust estimate of the topics for each document, we can average a large number of predictions for the same document:


```python
def average_predictions(model, test_document, num_trials=100):
    avg_preds = np.zeros((model.num_topics))
    for i in range(num_trials):
        avg_preds += model.predict(test_document, output_type='probability')[0]
    avg_preds = avg_preds/num_trials
    result = gl.SFrame({'topics':themes, 'average predictions':avg_preds})
    result = result.sort('average predictions', ascending=False)
    return result
```


```python
print average_predictions(topic_model, obama, 100)
```

    +---------------------+-------------------------------+
    | average predictions |             topics            |
    +---------------------+-------------------------------+
    |    0.594758064516   |        general politics       |
    |    0.13876344086    | American college and politics |
    |   0.0525268817204   |            Business           |
    |   0.0495161290323   |          team sports          |
    |   0.0395698924731   |      science and research     |
    |   0.0336290322581   |    international athletics    |
    |   0.0250537634409   |  Great Britain and Australia  |
    |   0.0238978494624   |       art and publishing      |
    |   0.0228225806452   |      international music      |
    |   0.0194623655914   |      music, TV, and film      |
    +---------------------+-------------------------------+
    [10 rows x 2 columns]
    
    

__Quiz Question:__ What is the topic most closely associated with the article about former US President George W. Bush? Use the average results from 100 topic predictions.


```python
bush = gl.SArray([wiki_docs[int(np.where(wiki['name']=='George W. Bush')[0])]])
print average_predictions(topic_model, bush, 100)
```

    +---------------------+-------------------------------+
    | average predictions |             topics            |
    +---------------------+-------------------------------+
    |    0.430087719298   |        general politics       |
    |    0.191432748538   | American college and politics |
    |    0.101608187135   |            Business           |
    |   0.0580409356725   |      science and research     |
    |   0.0494736842105   |       art and publishing      |
    |   0.0453801169591   |          team sports          |
    |   0.0372514619883   |  Great Britain and Australia  |
    |   0.0357602339181   |    international athletics    |
    |   0.0295614035088   |      music, TV, and film      |
    |   0.0214035087719   |      international music      |
    +---------------------+-------------------------------+
    [10 rows x 2 columns]
    
    

__Quiz Question:__ What are the top 3 topics corresponding to the article about English football (soccer) player Steven Gerrard? Use the average results from 100 topic predictions.


```python
gerrard = gl.SArray([wiki_docs[int(np.where(wiki['name']=='Steven Gerrard')[0])]])
print average_predictions(topic_model, gerrard, 100)
```

    +---------------------+-------------------------------+
    | average predictions |             topics            |
    +---------------------+-------------------------------+
    |       0.49012       |          team sports          |
    |       0.17116       |  Great Britain and Australia  |
    |       0.13144       |    international athletics    |
    |       0.03552       |      international music      |
    |       0.03264       |        general politics       |
    |       0.03168       |      music, TV, and film      |
    |        0.0294       |            Business           |
    |       0.02744       |       art and publishing      |
    |        0.0256       | American college and politics |
    |        0.025        |      science and research     |
    +---------------------+-------------------------------+
    [10 rows x 2 columns]
    
    

# Comparing LDA to nearest neighbors for document retrieval

So far we have found that our topic model has learned some coherent topics, we have explored these topics as probability distributions over a vocabulary, and we have seen how individual documents in our Wikipedia data set are assigned to these topics in a way that corresponds with our expectations. 

In this section, we will use the predicted topic distribution as a representation of each document, similar to how we have previously represented documents by word count or TF-IDF. This gives us a way of computing distances between documents, so that we can run a nearest neighbors search for a given document based on its membership in the topics that we learned from LDA. We can contrast the results with those obtained by running nearest neighbors under the usual TF-IDF representation, an approach that we explored in a previous assignment. 

We'll start by creating the LDA topic distribution representation for each document:


```python
wiki['lda'] = topic_model.predict(wiki_docs, output_type='probability')
```

Next we add the TF-IDF document representations:


```python
wiki['word_count'] = gl.text_analytics.count_words(wiki['text'])
wiki['tf_idf'] = gl.text_analytics.tf_idf(wiki['word_count'])
```

For each of our two different document representations, we can use GraphLab Create to compute a brute-force nearest neighbors model:


```python
wiki.head(2)
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">URI</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">text</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">lda</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Digby_Morrell&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Digby Morrell</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">digby morrell born 10<br>october 1979 is a former ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.037037037037,<br>0.529100529101, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Alfred_J._Lewy&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alfred J. Lewy</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">alfred j lewy aka sandy<br>lewy graduated from ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.432432432432,<br>0.0486486486486, ...</td>
    </tr>
</table>
<table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word_count</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">tf_idf</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'since': 1L, 'carltons':<br>1L, 'being': 1L, '2005': ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'since':<br>1.455376717308041, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'precise': 1L, 'thomas':<br>1L, 'closely': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'precise':<br>6.44320060695519, ...</td>
    </tr>
</table>
[2 rows x 6 columns]<br/>
</div>




```python
model_tf_idf = gl.nearest_neighbors.create(wiki, label='name', features=['tf_idf'],
                                           method='brute_force', distance='cosine')
model_lda_rep = gl.nearest_neighbors.create(wiki, label='name', features=['lda'],
                                            method='brute_force', distance='cosine')
```


<pre>Starting brute force nearest neighbors model training.</pre>



<pre>Starting brute force nearest neighbors model training.</pre>


Let's compare these nearest neighbor models by finding the nearest neighbors under each representation on an example document. For this example we'll use Paul Krugman, an American economist:


```python
model_tf_idf.query(wiki[wiki['name'] == 'Paul Krugman'], label='name', k=10)
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.00169288  | 15.6ms       |</pre>



<pre>| Done         |         | 100         | 408.208ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Elise Brezis</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.744498017262</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Maitreesh Ghatak</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.81564984831</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Kai A. Konrad</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.823700564406</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">David Colander</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.834625927759</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Richard Blundell</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.837934267874</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Gordon Rausser</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.83941534706</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Edward J. Nell</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.842178500015</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Robin Boadway</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.842374260596</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Tim Besley</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.843088109253</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10</td>
    </tr>
</table>
[10 rows x 4 columns]<br/>
</div>




```python
model_lda_rep.query(wiki[wiki['name'] == 'Paul Krugman'], label='name', k=10)
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.00169288  | 0us          |</pre>



<pre>| Done         |         | 100         | 15.6ms       |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.11022302463e-16</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">David Canter</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.00510122371062</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Bill Durodi%C3%A9</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.00624428797917</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Julian Cobbing</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.00749065226014</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">John Emmeus Davis</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0100786700829</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">David Benatar</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0100841246902</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Gabrielle Rifkind</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0104570623845</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Corey Robin</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0106040787042</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Faisal Devji</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0113324452423</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Cyril Belshaw</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0115265946965</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10</td>
    </tr>
</table>
[10 rows x 4 columns]<br/>
</div>



Notice that that there is no overlap between the two sets of top 10 nearest neighbors. This doesn't necessarily mean that one representation is better or worse than the other, but rather that they are picking out different features of the documents. 

With TF-IDF, documents are distinguished by the frequency of uncommon words. Since similarity is defined based on the specific words used in the document, documents that are "close" under TF-IDF tend to be similar in terms of specific details. This is what we see in the example: the top 10 nearest neighbors are all economists from the US, UK, or Canada. 

Our LDA representation, on the other hand, defines similarity between documents in terms of their topic distributions. This means that documents can be "close" if they share similar themes, even though they may not share many of the same keywords. For the article on Paul Krugman, we expect the most important topics to be 'American college and politics' and 'science and research'. As a result, we see that the top 10 nearest neighbors are academics from a wide variety of fields, including literature, anthropology, and religious studies.


__Quiz Question:__ Using the TF-IDF representation, compute the 5000 nearest neighbors for American baseball player Alex Rodriguez. For what value of k is Mariano Rivera the k-th nearest neighbor to Alex Rodriguez? (Hint: Once you have a list of the nearest neighbors, you can use `mylist.index(value)` to find the index of the first instance of `value` in `mylist`.)

__Quiz Question:__ Using the LDA representation, compute the 5000 nearest neighbors for American baseball player Alex Rodriguez. For what value of k is Mariano Rivera the k-th nearest neighbor to Alex Rodriguez? (Hint: Once you have a list of the nearest neighbors, you can use `mylist.index(value)` to find the index of the first instance of `value` in `mylist`.)


```python
AR = model_tf_idf.query(wiki[wiki['name'] == 'Alex Rodriguez'], label='name', k=5000)
AR
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.00169288  | 66.202ms     |</pre>



<pre>| Done         |         | 100         | 435.814ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alex Rodriguez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alex Rodriguez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alex Rodriguez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">John Rodriguez (baseball)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.643662221072</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alex Rodriguez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alfonso Soriano</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.716766505271</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alex Rodriguez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Eladio Rodriguez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.727952380269</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alex Rodriguez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Maggie Rodriguez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.730077138491</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alex Rodriguez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Derek Jeter</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.744062022625</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alex Rodriguez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Felipe Rodriguez (soccer)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.750710450705</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alex Rodriguez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Rich Rodriguez (baseball)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.755699161545</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alex Rodriguez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Josh Hamilton</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.758732675823</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alex Rodriguez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Chris Davis (baseball)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.765630370531</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10</td>
    </tr>
</table>
[5000 rows x 4 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>




```python
AR[AR['reference_label'] == 'Mariano Rivera']
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alex Rodriguez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Mariano Rivera</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.835848608513</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">53</td>
    </tr>
</table>
[? rows x 4 columns]<br/>Note: Only the head of the SFrame is printed. This SFrame is lazily evaluated.<br/>You can use sf.materialize() to force materialization.
</div>




```python
AR_lda = model_lda_rep.query(wiki[wiki['name'] == 'Alex Rodriguez'], label='name', k=5000)
AR_lda
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.00169288  | 0us          |</pre>



<pre>| Done         |         | 100         | 46.2ms       |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alex Rodriguez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alex Rodriguez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alex Rodriguez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Jos%C3%A9 Capell%C3%A1n</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.00303427483404</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alex Rodriguez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Bruce Mulherin</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.00356776022103</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alex Rodriguez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Bart Crashley</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0037768556863</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alex Rodriguez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Jeff Conine</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.00384781626279</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alex Rodriguez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Josh Hamilton</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0042253248535</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alex Rodriguez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Toby Borland</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.00456753816294</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alex Rodriguez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Frank Tanana</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.00466292479962</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alex Rodriguez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Rene Bourque</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.00466629646998</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alex Rodriguez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Bruce Dreckman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.00471831332232</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10</td>
    </tr>
</table>
[5000 rows x 4 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>




```python
AR_lda[AR_lda['reference_label'] == 'Mariano Rivera']
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alex Rodriguez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Mariano Rivera</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0124614810028</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">722</td>
    </tr>
</table>
[? rows x 4 columns]<br/>Note: Only the head of the SFrame is printed. This SFrame is lazily evaluated.<br/>You can use sf.materialize() to force materialization.
</div>



# Understanding the role of LDA model hyperparameters

Finally, we'll take a look at the effect of the LDA model hyperparameters alpha and gamma on the characteristics of our fitted model. Recall that alpha is a parameter of the prior distribution over topic weights in each document, while gamma is a parameter of the prior distribution over word weights in each topic. 

In the video lectures, we saw that alpha and gamma can be thought of as smoothing parameters when we compute how much each document "likes" a topic (in the case of alpha) or how much each topic "likes" a word (in the case of gamma). In both cases, these parameters serve to reduce the differences across topics or words in terms of these calculated preferences; alpha makes the document preferences "smoother" over topics, and gamma makes the topic preferences "smoother" over words.

Our goal in this section will be to understand how changing these parameter values affects the characteristics of the resulting topic model.

__Quiz Question:__ What was the value of alpha used to fit our original topic model? 

alpha                          : 5.0
beta                           : 0.1

__Quiz Question:__ What was the value of gamma used to fit our original topic model? Remember that GraphLab Create uses "beta" instead of "gamma" to refer to the hyperparameter that influences topic distributions over words.

We'll start by loading some topic models that have been trained using different settings of alpha and gamma. Specifically, we will start by comparing the following two models to our original topic model:
 - tpm_low_alpha, a model trained with alpha = 1 and default gamma
 - tpm_high_alpha, a model trained with alpha = 50 and default gamma


```python
tpm_low_alpha = gl.load_model('topic_models/lda_low_alpha')
tpm_high_alpha = gl.load_model('topic_models/lda_high_alpha')
```

### Changing the hyperparameter alpha

Since alpha is responsible for smoothing document preferences over topics, the impact of changing its value should be visible when we plot the distribution of topic weights for the same document under models fit with different alpha values. In the code below, we plot the (sorted) topic weights for the Wikipedia article on Barack Obama under models fit with high, original, and low settings of alpha.


```python
a = np.sort(tpm_low_alpha.predict(obama,output_type='probability')[0])[::-1]
b = np.sort(topic_model.predict(obama,output_type='probability')[0])[::-1]
c = np.sort(tpm_high_alpha.predict(obama,output_type='probability')[0])[::-1]
ind = np.arange(len(a))
width = 0.3

def param_bar_plot(a,b,c,ind,width,ylim,param,xlab,ylab):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    b1 = ax.bar(ind, a, width, color='lightskyblue')
    b2 = ax.bar(ind+width, b, width, color='lightcoral')
    b3 = ax.bar(ind+(2*width), c, width, color='gold')

    ax.set_xticks(ind+width)
    ax.set_xticklabels(range(10))
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    ax.set_ylim(0,ylim)
    ax.legend(handles = [b1,b2,b3],labels=['low '+param,'original model','high '+param])

    plt.tight_layout()
    
param_bar_plot(a,b,c,ind,width,ylim=1.0,param='alpha',
               xlab='Topics (sorted by weight of top 100 words)',ylab='Topic Probability for Obama Article')
```


![png](output_63_0.png)


Here we can clearly see the smoothing enforced by the alpha parameter - notice that when alpha is low most of the weight in the topic distribution for this article goes to a single topic, but when alpha is high the weight is much more evenly distributed across the topics.

__Quiz Question:__ How many topics are assigned a weight greater than 0.3 or less than 0.05 for the article on Paul Krugman in the **low alpha** model?  Use the average results from 100 topic predictions.


```python
krug = gl.SArray([wiki_docs[int(np.where(wiki['name']=='Paul Krugman')[0])]])
krug_100 = average_predictions(tpm_low_alpha, krug, 100)
krug_100
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">average predictions</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">topics</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.628333333333</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">art and publishing</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.213024691358</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">music, TV, and film</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0624074074074</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">international music</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0162345679012</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Great Britain and<br>Australia ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0157407407407</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">international athletics</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0153086419753</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">American college and<br>politics ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0137654320988</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">team sports</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0137037037037</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">science and research</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0109259259259</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">general politics</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0105555555556</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Business</td>
    </tr>
</table>
[10 rows x 2 columns]<br/>
</div>




```python
a = krug_100['average predictions']
print len([x for x in a if x>0.3 or x < 0.05]) 
```

    8
    

__Quiz Question:__ How many topics are assigned a weight greater than 0.3 or less than 0.05 for the article on Paul Krugman in the **high alpha** model? Use the average results from 100 topic predictions.


```python
krug_100_high = average_predictions(tpm_high_alpha, krug, 100)
krug_100_high
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">average predictions</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">topics</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.322777777778</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Great Britain and<br>Australia ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.188015873016</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">international athletics</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0918253968254</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">art and publishing</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0604365079365</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Business</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0602777777778</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">music, TV, and film</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0597222222222</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">team sports</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0596031746032</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">international music</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0594047619048</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">American college and<br>politics ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.050753968254</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">general politics</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0471825396825</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">science and research</td>
    </tr>
</table>
[10 rows x 2 columns]<br/>
</div>




```python
b = krug_100_high['average predictions']
print len([x for x in b if x>0.3 or x < 0.05]) 
```

    2
    

### Changing the hyperparameter gamma

Just as we were able to see the effect of alpha by plotting topic weights for a document, we expect to be able to visualize the impact of changing gamma by plotting word weights for each topic. In this case, however, there are far too many words in our vocabulary to do this effectively. Instead, we'll plot the total weight of the top 100 words and bottom 1000 words for each topic. Below, we plot the (sorted) total weights of the top 100 words and bottom 1000 from each topic in the high, original, and low gamma models.

Now we will consider the following two models:
 - tpm_low_gamma, a model trained with gamma = 0.02 and default alpha
 - tpm_high_gamma, a model trained with gamma = 0.5 and default alpha


```python
del tpm_low_alpha
del tpm_high_alpha
tpm_low_gamma = gl.load_model('topic_models/lda_low_gamma')
tpm_high_gamma = gl.load_model('topic_models/lda_high_gamma')
```


```python
a_top = np.sort([sum(tpm_low_gamma.get_topics(topic_ids=[i], num_words=100)['score']) for i in range(10)])[::-1]
b_top = np.sort([sum(topic_model.get_topics(topic_ids=[i], num_words=100)['score']) for i in range(10)])[::-1]
c_top = np.sort([sum(tpm_high_gamma.get_topics(topic_ids=[i], num_words=100)['score']) for i in range(10)])[::-1]

a_bot = np.sort([sum(tpm_low_gamma.get_topics(topic_ids=[i], num_words=547462)[-1000:]['score']) for i in range(10)])[::-1]
b_bot = np.sort([sum(topic_model.get_topics(topic_ids=[i], num_words=547462)[-1000:]['score']) for i in range(10)])[::-1]
c_bot = np.sort([sum(tpm_high_gamma.get_topics(topic_ids=[i], num_words=547462)[-1000:]['score']) for i in range(10)])[::-1]

ind = np.arange(len(a))
width = 0.3
    
param_bar_plot(a_top, b_top, c_top, ind, width, ylim=0.6, param='gamma',
               xlab='Topics (sorted by weight of top 100 words)', 
               ylab='Total Probability of Top 100 Words')

param_bar_plot(a_bot, b_bot, c_bot, ind, width, ylim=0.0002, param='gamma',
               xlab='Topics (sorted by weight of bottom 1000 words)',
               ylab='Total Probability of Bottom 1000 Words')
```


![png](output_73_0.png)



![png](output_73_1.png)


From these two plots we can see that the low gamma model results in higher weight placed on the top words and lower weight placed on the bottom words for each topic, while the high gamma model places relatively less weight on the top words and more weight on the bottom words. Thus increasing gamma results in topics that have a smoother distribution of weight across all the words in the vocabulary.

__Quiz Question:__ For each topic of the **low gamma model**, compute the number of words required to make a list with total probability 0.5. What is the average number of words required across all topics? (HINT: use the get\_topics() function from GraphLab Create with the cdf\_cutoff argument).


```python
tpm_low_gamma.get_topics(topic_ids=[0], cdf_cutoff=0.5)
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">topic</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">score</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">born</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0199425183958</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">years</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0112323096805</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">time</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0107843438708</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">south</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.00903738412603</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2005</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.00892833278333</td>
    </tr>
</table>
[5 rows x 3 columns]<br/>
</div>




```python
low_gamma_all_topics = [len(tpm_low_gamma.get_topics(topic_ids=[i], cdf_cutoff=0.5)) for i in range(10)]
low_gamma_all_topics
```




    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]




```python
sum(low_gamma_all_topics) / float(len(low_gamma_all_topics))
```




    5.0



__Quiz Question:__ For each topic of the **high gamma model**, compute the number of words required to make a list with total probability 0.5. What is the average number of words required across all topics? (HINT: use the get\_topics() function from GraphLab Create with the cdf\_cutoff argument).


```python
tpm_high_gamma.get_topics(topic_ids=[0], cdf_cutoff=0.5)
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">topic</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">score</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">league</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0153348831959</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">season</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0144899511253</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">played</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0141521274466</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">team</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0104718628672</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">football</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.00916904174868</td>
    </tr>
</table>
[5 rows x 3 columns]<br/>
</div>




```python
high_gamma_all_topics = [len(tpm_high_gamma.get_topics(topic_ids=[i], cdf_cutoff=0.5)) for i in range(10)]
high_gamma_all_topics
```




    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]



We have now seen how the hyperparameters alpha and gamma influence the characteristics of our LDA topic model, but we haven't said anything about what settings of alpha or gamma are best. We know that these parameters are responsible for controlling the smoothness of the topic distributions for documents and word distributions for topics, but there's no simple conversion between smoothness of these distributions and quality of the topic model. In reality, there is no universally "best" choice for these parameters. Instead, finding a good topic model requires that we be able to both explore the output (as we did by looking at the topics and checking some topic predictions for documents) and understand the impact of hyperparameter settings (as we have in this section).
