""" uses the paraphrase_mining method with a pre-trained SBERT model to calculate 
    the semantic similarity between abstracts from english wikipedia articles. 
    time taken: 0hr:25 on colab GPU """

from sentence_transformers import SentenceTransformer, util
import csv
import pickle
import time

# load pre-trained SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# extract data from csv file to list
data = []
with open("cleaned_data/abstracts_over150chars.txt") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader:
        data.append(row[2])

# get list of similar pairs of abstracts ranked by the cosine similarity of their vectors
start_time = time.time()
paraphrases = util.paraphrase_mining(model, data, 
                                     corpus_chunk_size=int(len(data)/32),
                                     query_chunk_size=2000, 
                                     top_k=6, 
                                     show_progress_bar=True)

# display elapsed time 
end_time = time.time()
print("time taken: ", end_time - start_time)

# save list of similar pairs
with open("models/SBERT/over150chars/paraphrase_pairs_6deep.pkl", "wb") as file:
    pickle.dump(paraphrases, file)

sample_index = 0

# iterate through list of pairs to find the indexes of similar abstracts
for pair in paraphrases:
    score, i, j = pair
    if int(i) == sample_index:
        print('aaa')
        # display relevant title and abstract from dataset using index
        print(f"{data[j][0]} \n{data[j][1]} \n{data[j][2]} \n[score: {score :.3f}] \n")
    elif int(j) == sample_index:
        print('bbb')
        # display relevant title and abstract from dataset using index
        print(f"{data[i][0]} \n{data[i][1]} \n{data[i][2]} \n[score: {score :.3f}] \n")

