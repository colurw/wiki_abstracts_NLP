""" Searches for clusters of articles in the SBERT embeddings, using a non-linear
    dimensionality reduction method (UMAP) followed by a density-based clustering 
    method (HBDSCAN).  Significant words are calculated using TF-IDF for each cluster.  """ 


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
import hdbscan
import umap.umap_ as umap
import statistics
import imports.c_TF_IDF as tfidf
import pandas as pd

# load numpy array of sbert embeddings
with open("models/SBERT/over150chars/embeddings_nump_nrml.pkl", mode="rb") as file:
    embeddings = pickle.load(file)
print('loaded')

# create subset of embeddings for hyperparameter tuning
embeddings_subset = embeddings[0:30000]

# assess explained variance using PCA, to guide selection of UMAP's 'n_components' hyperparameter
pca = PCA().fit(embeddings)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.savefig(f'umap_hyperparameter_tuning/explained_variance')

## >>> 80% of embeddings variance explained by 130 dimensions, and 90% by 200.

# assess optimum value of 'n_neighbours' hyperparameter in UMAP by passing list possible values
for n in (5, 15, 45, 135):
    umap_test = umap.UMAP(n_neighbors=n, 
                        n_components=2,
                        min_dist=0.0, 
                        metric='cosine').fit_transform(embeddings_subset)
    results = pd.DataFrame(umap_test, columns=['x', 'y'])

    # draw 3D graph of UMAP embeddings for current value of 'n'
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)  # , projection='3d'
    ax.scatter(umap_test[:,0], umap_test[:,1], s=1)
    ax.set_title(f'n={n}')
    plt.savefig(f'umap_hyperparameter_tuning/n={n}_2d')

# ## >>> n_neighbours =< 15 seems best ?

# create subset of embeddings to prevent OOM error (300000 articles needs >70Gb RAM during TF-IDF)
embeddings = embeddings[0:300000]

# create umapper object, then reduce dimensionality of embeddings
umapper = umap.UMAP(n_neighbors=5,           # 5,10,10
                    n_components=120,        # 160,120,120
                    min_dist=0.0,
                    set_op_mix_ratio=1,      # 1,1,0.5       
                    metric='cosine')     
umap_embeddings = umapper.fit_transform(embeddings)
print('umapped')

# create hierarchical clusterer object, to find groups of embeddings in close proximity to each other
clusterer = hdbscan.HDBSCAN(min_cluster_size=10,     # 15,5,5
                            metric='euclidean',      # embeddings normalised
                            cluster_selection_method='eom')
clusterer.fit(umap_embeddings)
print('clustered')

# save output
with open('clusterer_3.pkl', 'wb') as file:
    pickle.dump(clusterer, file)

with open('clusterer_3.pkl', 'rb') as file:
    clusterer = pickle.load(file)

# check output
print(len(clusterer.labels_), 'articles processed')
print(max(clusterer.labels_), 'clusters found')
print(len([label for label in clusterer.labels_ if label == -1]), 'articles not clustered')
most_common_label = statistics.mode([label for label in clusterer.labels_ if label != -1])
print(len([label for label in clusterer.labels_ if label == most_common_label]), 'size of largest cluster')

# read all csv data to pandas dataframe
pd.options.mode.copy_on_write = True
df = pd.read_csv("cleaned_data/abstracts_over150chars.txt", header=None, index_col=None)
df.columns =["index", "title", "abstract"]
print(df.head(30))

# read cluster labels to pandas dataframe, concatenate with df
df_labels = pd.DataFrame({'cluster': clusterer.labels_})
df = pd.concat([df, df_labels], axis=1)
print(df.head(30))

# drop articles not processed during clustering ('cluster' == NaN), and unneccesary columns
df.dropna(subset = ["cluster"], inplace=True)  # dodgy?
df.drop(columns=["index", "title"], axis=1, inplace=True) 

# aggregate abstracts by cluster label
df_agg = df.groupby(["cluster"], as_index=False, sort=True).agg({'abstract': ' '.join})
print(df_agg.head(10))

# calculate TF-IDF (text frequency-inverse document frequency) for each cluster against all clusters
tf_idf, count = tfidf.c_tf_idf(documents=df_agg.abstract.values, 
                             m=len(df), 
                             ngram_range=(1, 1))    # length of keyword groups

# create dictionary of keywords sorted by cluster_id
keywords_dict = tfidf.extract_top_n_words_per_topic(tf_idf, count, 
                                                    docs_by_topic=df_agg, 
                                                    n=6)

# save dictionaries
with open('keywords_dict_new.pkl', 'wb') as file:
    pickle.dump(keywords_dict, file)

