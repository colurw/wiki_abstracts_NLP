""" uses a pre-trained universal sentence encoder model to calculate 
    semantic similarity between abstracts from english wikipedia articles.  
    time taken: 0hr:08 """

import tensorflow as tf
import tensorflow_hub as hub
from absl import logging
import numpy as np
import csv
import time
import pickle

# load pre_trained USE module from TF Hub
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)

# reduce logging output.
logging.set_verbosity(logging.ERROR)

# process embeddings in 32 batches to prevent OOM error
embeddings =[]
batch_size = 38233
for i in range(1):

    # extract single batch of abstracts from cleaned csv file
    abstracts = []
    with open("cleaned_data/abstracts_test.txt") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for j, row in enumerate(csv_reader):
            if j >= batch_size * i and j < batch_size * (i+1):
                abstracts.append(row[2])

    # embed abstracts to list of vectors
    start_time = time.time()
    current_embeddings = model(abstracts)

    # add embeddings to master list
    embeddings.extend(current_embeddings)

# display elapsed time 
end_time = time.time()
print("time taken: ", end_time - start_time)

# convert to numpy array
embeddings = np.array(embeddings).astype(np.float32)

# save embedded abstracts
with open(f"models/USE/over150chars/all_embeddingsxxxx.pkl", "wb") as file:
    pickle.dump(embeddings, file)

# check all embeddings were captured
print(embeddings.shape)




