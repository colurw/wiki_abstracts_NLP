# Wiki_Abstracts_NLP 

Explores the functionality of several of Python's Natural
Language Processing libraries and large language models.  It compares their 
proficiency at discovering clusters of semantically-similar documents, and their
ability to perform unsupervised topic modelling - _i.e._ applying a human-understandable 
short descriptive label to these document clusters.

## 1_data_cleaning.py  

Parses XML dumps of English Wikipedia abstracts (the first 
paragraph of an article) from wikipedia.org and creates a CSV file containing 
~1.2M entries, consisting of [index number, title, abstract].  
List articles, recurrent events, and articles with abstracts shorter than 150 
characters are excluded.

## 2x_[model]_train/embed/etc.py 

Creates document-level embeddings of each article.  
Three models and their libraries are compared: Sentence_Transformers' Sentence-BERT, 
Google's Universal Sentence Encoder, and Gensim's Doc2Vec.  
The first two embed the articles' vectors using a pre-trained model, whilst Doc2Vec 
first trains a model on the provided corpus prior to embedding the articles.

## 3_spaCy_entity_recognition.py  

Attempts to classify the article title into one of 
eighteen categories using the EntityRecogniser module from SpaCy and a fuzzy word-
matching algorithm, in order to infer the context of the title with respect to the 
abstract, helping it to distinguish, say, Apple [ORG] from Apple [PRODUCT].

## 4_H-cluster_keywords.py  

Applies a Text Frequency-Inverse Document Frequency (TF-IDF)
calculation on words found per cluster of semantically-similar articles.  
The clusters are generated using a non-linear dimensionality reduction method (UMAP) 
followed by a hierachical clustering method (HBDSCAN), in the hopes of preserving 
as much higher dimension structure as possible, and avoiding the downsides of 
k-means clustering, such as forcing outliers into clusters, and having to guess the 
total number of clusters beforehand.  

Words and phrases found by the TF-IDF method can be considered as keywords (or 
rarely, descriptors) of a topic - rather than true topic labels.

## 5_chatGPT_topic_labelling.py  

Attempts to label topic clusters with a definitive 
name.  Traditionally this is/was a difficult problem, when labelled training 
examples are not available.

## 6_everything_all_at_once.py  

Uses all of the above methods, embeddings and models 
to generate a text file output that (ideally) contains relevant information, such as 
, entity type, semantically-similar articles, keywords, and topic labels.  
These text files can be found in the /examples folder.

## Findings

Doc2Vec performs worst at finding semantically similar articles, although this perhaps
isn'tsurprising given the relatively small training corpus.  The pre-trained transformer
models do much better at the task, with USE taking a significantly longer time than SBERT, 
whilst producing typically inferior results.

SpaCy does OK at topic labelling, whilst Chat-GPT performs exceptionally well - after some 
basic prompt engineering.  TF-IDF, despite its more rudimentary nature, also provides 
useful results.






