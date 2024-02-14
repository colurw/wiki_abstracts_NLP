# Wiki_Abstracts_NLP 

Explores the functionality of several of Python's Natural
Language Processing (NLP) libraries and large language models.  It compares their 
proficiency at discovering clusters of semantically similar documents, and their
ability to perform unsupervised topic modelling - _i.e._ applying a human-understandable 
short descriptive label to these document clusters.

## 1_data_cleaning.py  

Parses XML dumps of English Wikipedia abstracts (the first 
paragraph of an article) from wikipedia.org and creates a CSV file containing 
~1.2M entries, consisting of [index number, title, abstract].  
List articles, recurrent events, and articles with abstracts shorter than 150 
characters are excluded.

## 2x_[model]_tokenize_train_embed.py 

Creates document-level embeddings of each article.  
Three models and their libraries are compared: Sentence_Transformers' Sentence-BERT, 
Google's Universal Sentence Encoder, and Gensim's Doc2Vec.  
The first two embed the article using a pre-trained model, whilst Doc2Vec 
first trains a model on the provided corpus prior to embedding the articles.

## 3_spaCy_entity_recognition.py  

Attempts to classify the article title into one of 
eighteen categories using the EntityRecogniser module from SpaCy and a fuzzy word-matching algorithm, in order to infer the context of the title with respect to the 
abstract, helping it to distinguish, say, Apple [ORG] from Apple [PRODUCT].

## 4_H-cluster_keywords.py  

Applies a Text Frequency-Inverse Document Frequency (TF-IDF)
calculation on words found per cluster of semantically-similar articles in the SBERT embedding.  

The clusters are found using a non-linear dimensionality reduction method (UMAP) 
followed by a hierarchical clustering method (HDBSCAN).  The first step is necessary to limit the 
quadratic time complexity of the second step.

Using HDBSCAN avoids the downsides of k-means clustering, such as forcing outliers into clusters, 
and having to guess the total number of clusters beforehand.  It is also able to preserve more of
the higher dimension structure of the data by 'compressing' empty regions in the embedding-space.

Words and phrases found by the TF-IDF method can be considered as keywords (or 
rarely, descriptors) of a topic - rather than true topic labels.

## 5_chatGPT_topic_labelling.py  

Attempts to label the topic clusters with a definitive 
name.  Traditionally this is/was a difficult problem if labelled training 
examples are not available.

## 6_everything_all_at_once.py  

Uses all the above methods, embeddings and models 
to generate a text file output that contains relevant information, such as 
, entity type, semantically-similar articles, keywords, and topic labels.  
These text files can be found in the /examples folder.

## Findings

Doc2Vec performs the worst at finding semantically similar articles, although this perhaps
isn't surprising given the relatively small training corpus in this case.  The pre-trained transformer
models do much better at the task, with USE taking a significantly longer time than SBERT, 
whilst producing mostly similar results.  
The difference may be due to the optimised 'cosine similarity' search function
included with the SBERT library compared to my implementation of it using Tensorflow, or due to the time complexity of running the 
search on a 384 dimension embedding, compared to one with 512 dimensions.  

SpaCy's entity recognition performance appears mediocre, but would very likely improve by choosing to download and use a larger language model.

Chat-GPT performs exceptionally well, after some basic prompt engineering.  TF-IDF, despite its more rudimentary nature, also provides 
useful, if less precise, results.


## More

Attempts at applying these techniques to the latent space of a Large Language Model can be found [here](https://github.com/colurw/pandora_NLP).





