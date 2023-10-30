""" Finds semantically similar articles by comparing their vectorised
    abstracts with a vector inferred from a previously unseen text string"""

from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize
import csv
import random

# extract data to list
data = []
with open("cleaned_data/PV/abstracts_over150chars.txt") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader:
        data.append(row)

string = "Surveillance centre hailed as critical in capture of escaped terror suspect."
tokens = word_tokenize(string.lower())
model = Doc2Vec.load("models/PV/over150chars_200vec/d2v.model")
inferred_vector = model.infer_vector(tokens)

similar_articles = model.dv.most_similar([inferred_vector], topn=len(model.dv))
number_of_matches = 10
for rank in range(number_of_matches):
    index, score = (similar_articles[rank])
    print(data[int(index)][1].upper())
    print(data[int(index)][2])
    print("[score: " + "%.3f" % score + "]")
    print()



# display similar articles
# similar_doc = model.dv.most_similar(random_index)
# number_of_matches = 5
# for rank in range(number_of_matches):
#     index, score = (similar_doc[rank])
#     print(data[int(index)][1].upper())
#     print(data[int(index)][2])
#     print("[score: " + "%.3f" % score + "]")
#     print()

