from sentence_transformers import SentenceTransformer
import csv
import pickle
import time

# load pre-trained SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# extract data from csv file to list
abstracts = []
with open("cleaned_data/abstracts_over150chars.txt") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader:
        abstracts.append(row[2])

#Compute embeddings
start_time = time.time()
embeddings = model.encode(abstracts, 
                          convert_to_numpy = True,       # allows downstream gpu or cpu similarity method
                          show_progress_bar = True, 
                          normalize_embeddings = True)   # allows faster dot product similarity method

# display elapsed time 
end_time = time.time()
print("time taken: ", end_time - start_time)

# save abstract embeddings
with open("models/SBERT/over150chars/embeddings_nump_nrml.pkl", "wb") as file:
    pickle.dump(embeddings, file)

