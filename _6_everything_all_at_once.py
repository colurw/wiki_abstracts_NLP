""" Finds semantically similar wikipedia articles by comparing their vectorised 
    abstracts (USE model), or by getting data from a previously generated 
    look-up table (SBERT & D2V models).    
    SpaCy attempts to categorise the article into one of eighteen broad categories
    using a fuzzy matching algorithm, whilst ChatGPT attempts to infer a more 
    fine-grained topic label by analysing the two groups of semantically similar 
    articles. 
    TF-IDF keywords from the sample article's HBDSCAN cluster are retrieved from 
    dictionaries. """


from gensim.models.doc2vec import Doc2Vec
from sentence_transformers import util
from _3_spaCy_entity_recognition import entity_recognition_fuzzy_match
from _5_chatGPT_topic_labelling import chatGPT_cluster_label
import tensorflow as tf
import csv
import random
import pickle
import time
import textwrap


def fprint(file, my_string, width=100):
    """ wraps a string and prints as paragraph to a text file """
    if len(my_string) > width:
        my_string = textwrap.fill(my_string, width, fix_sentence_endings=True)
    
    with open(f'examples/{file}.txt', 'a') as file:
        file.write(my_string + '\n')
    
    return my_string


def display_D2V_most_similar(similar_articles, top_k, file):
    """ prints title and abstract from 'top_k' articles found by Doc2Vec.most_similar """
    
    fprint(title, "\n >>>> Semantically Similar Articles (Doc2Vec): \n")
    for rank in range(top_k):
        index, score = (similar_articles[rank])  
        
        # get relevant title and abstract from dataset
        fprint(file, f"{data[int(index)][1].upper()}")
        fprint(file, f" {data[int(index)][2]}")
        fprint(file, "[score: " + "%.3f" % score + "] \n")


def find_and_display_SBERT_most_similar(target, all_embeddings, top_k, file):
    """ prints title and abstract from 'top_k' articles found by SBERT.semantic_search """
    
    articles = []
    # use util.semantic_search to calculate cosine similarities 
    hits = util.semantic_search(target, all_embeddings, top_k=top_k+1)
    # get relevant titles and abstracts from dataset
    hits = hits[0]
    
    # format results for chatgpt topic labeling function
    for hit in hits[1:top_k+1]:
        article = f"{data[hit['corpus_id']][1]} " + f"{data[hit['corpus_id']][2]}"
        articles.append(article)
    
    # display topic label, relevant titles and abstracts from dataset
    fprint(file, f"\n\n>>>> ChatGPT-SentenceBERT Cluster Label:  {chatGPT_cluster_label(articles)}\n")
    for hit in hits[1:top_k+1]:
        fprint(file, f"{data[hit['corpus_id']][1].upper()}")
        fprint(file, f"{data[hit['corpus_id']][2]}") 
        fprint(file, f"[score: {(hit['score']) :.3f}] \n")


def find_and_display_USE_most_similar(target, all_embeddings, top_k, file):
    """ prints title and abstract from 'top_k' articles found by keras cosine_similarity """
    articles = []
    # calculate cosine similarities
    cosine_similarities = tf.keras.losses.cosine_similarity(target, all_embeddings).numpy() * -1
    best_values, best_indices = tf.math.top_k(cosine_similarities, k=top_k+1)
    
    # format results for chatgpt topic labeling function
    for i in range(1, top_k+1):
        article = f"{data[best_indices[i]][1]} " + f"{data[best_indices[i]][2]}"
        articles.append(article)
    
    # display topic label, relevant titles and abstracts from dataset
    fprint(file, f"\n>>>> ChatGPT-USE Cluster Label:  {chatGPT_cluster_label(articles)}\n")
    for i in range(1, top_k+1):
        fprint(file, f"{data[best_indices[i]][1].upper()}")
        fprint(file, f" {data[best_indices[i]][2]}")
        fprint(file, f"[semantic similarity: {best_values[i] :.3f}] \n")


def get_keywords(sample_index, clusterer_object, keywords_dict, n):
    """ returns the top 'n' keywords from a given article's cluster """
    keywords = []
    # lookup cluster_id
    cluster_id = clusterer_object.labels_[sample_index]
    
    if int(cluster_id) == -1:
        keywords = ' Niche article - no cluster found'

        return keywords
    
    else:
        # lookup keywords from dictionary using cluster_id
        for keyword, score in keywords_dict[cluster_id][0:n]:
            keywords.append(keyword)
        
        return keywords


# load pickled files from training, embeddings, and keyword analysis
print('loading models...')

with open("models/SBERT/over150chars/embeddings_nump_nrml.pkl", mode="rb") as file:
    SBERT_embeddings = pickle.load(file)

with open("models/USE/over150chars/all_embeddings.pkl", mode="rb") as file:
    USE_embeddings = pickle.load(file)

with open('keywords_dict.pkl', 'rb') as file:
    keywords_dict = pickle.load(file)

with open('keywords_dict_2.pkl', 'rb') as file:
    keywords_dict_2 = pickle.load(file)

with open('clusterer_3.pkl', 'rb') as file:
    clusterer_object = pickle.load(file)

d2vmodel = Doc2Vec.load("models/D2V/over150chars_300vec/d2v.model")

# extract index-title-abstract data from cleaned csv file
data = []
with open("cleaned_data/abstracts_over150chars.txt") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader:
        data.append(row)

# limit search range to 80000 to find articles within range of 2-word TF-IDF calculation
LIMIT = 80000    

# select article from dataset by index number
over_ride = input("\n >>> Enter article number, or hit ENTER for random choice: ")
if over_ride != '':
    sample_index = int(over_ride)
    reps = 1
else:
    # choose random article from dataset
    sample_index = random.randint(0, LIMIT)      
    reps = int(input("\n >>> Enter amount of random articles to find: "))

# loop find n articles
for rep in range(reps):
    if reps != 1:
        sample_index = random.randint(0, LIMIT)
        print(sample_index)

    # get article data
    for row in data:
        if int(row[0]) == sample_index:
            title = str(row[1])
            abstract = str(row[2])
            fprint(title, f"{row[0]}\n")
            fprint(title, title.upper())
            fprint(title, f" {abstract}")

    # determine category using spaCy entity recogniton
    label, description = entity_recognition_fuzzy_match(abstract, title)
    fprint(title, f"\n>>>> SpaCy Entity Recognition:   {label} - {description}\n")

    # look up keywords for article cluster
    keywords = get_keywords(sample_index, clusterer_object, keywords_dict, n=5)
    fprint(title, f">>>> H-Cluster Keywords:   {str(keywords)}")

    keywords = get_keywords(sample_index, clusterer_object, keywords_dict_2, n=3)
    fprint(title, f">>>> H-Cluster Keywords:   {str(keywords)}")

    # find number of articles with same cluster id
    cluster_id = clusterer_object.labels_[sample_index]
    cluster_size = len([value for value in clusterer_object.labels_ if value == cluster_id])
    if cluster_id == -1:
        cluster_size = 'n/a'
    fprint(title, f">>>> Hierarchical Cluster Size:   {str(cluster_size)}")

    # find similar articles amongst previously-generated SBERT embeddings using cosine similarity
    start_time = time.time()
    find_and_display_SBERT_most_similar(SBERT_embeddings[sample_index], SBERT_embeddings, 4, title)
    end_time = time.time()
    sbert_time = end_time - start_time

    # find similar articles amongst previously-generated USE embeddings using cosine similarity
    start_time = time.time()
    find_and_display_USE_most_similar(USE_embeddings[sample_index], USE_embeddings, 4, title)
    end_time = time.time()
    use_time = end_time - start_time
    
    # find similar articles amongst previously-generated D2V embeddings using cosine similarity
    start_time = time.time()
    similar_articles = d2vmodel.dv.most_similar(sample_index) 
    display_D2V_most_similar(similar_articles, 4, title)
    end_time = time.time()
    d2v_time = end_time - start_time

    # display time taken during each search
    fprint(title, f"\n>>>> Time taken (SBERT): {sbert_time :.1f} sec")
    fprint(title, f">>>> Time taken (USE): {use_time :.1f} sec")
    fprint(title, f">>>> Time taken (D2V): {d2v_time :.1f} sec")

