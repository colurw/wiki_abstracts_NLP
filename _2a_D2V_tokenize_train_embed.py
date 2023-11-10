""" creates a Doc2Vec paragraph-vector model of abstracts from english wikipedia articles. 
    time taken: 0hr:41 """

import gensim
from gensim.models.doc2vec import TaggedDocument # , Doc2Vec
import nltk
from nltk.tokenize import word_tokenize
import csv
import time
import logging

# extract data to list
data = []
with open("cleaned_data/abstracts_over150chars.txt") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader:
        data.append(row[2])

nltk.download("punkt")
start_time = time.time()

# create indexed and tokenised data
tagged_data = [TaggedDocument(words=word_tokenize(abstract.lower()), tags=[str(index)]) 
               for index, abstract in enumerate(data)]

# create paragraph vector model
model = gensim.models.doc2vec.Doc2Vec(vector_size=300,  # number of dimensions per vector (100-300)
                                      min_count=4,      # ignore words with fewer occurences (2-5)
                                      epochs=15,        # training epochs (10-20)
                                      dm=1)             # PV-DM or PV-DBOW

# train model
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
model.save("models/D2V/over150chars_300vec/d2v.model")

# display elapsed time 
end_time = time.time()
print("time taken: ", end_time - start_time)
