""" Category prediction based on entity recognition of all words in, 
    and in the context of, the abstract. Article title is the fuzzy-
    matched against words in abstract to get relevant category label """

import spacy 
import csv
import random
import difflib


def entity_recognition_fuzzy_match(abstract, title):
    """ natural language processing using spaCy standard english model and fuzzy entity matching"""
    # >>> python -m spacy download en_core_web_sm
    nlp = spacy.load('en_core_web_sm')  
    doc = nlp(abstract)
    # extract entities from abstract, save them as dictionary keys with their labels as values
    entities = [(str(entity), entity.label_) for entity in doc.ents]
    entities_dict = dict(entities)
    # search dictionary keys for a match with article title, lowering matching criteria with each cycle
    for step in range(100,0,-1):
        matches_list = difflib.get_close_matches(word = title, 
                                                possibilities = [key for key in entities_dict.keys()], 
                                                cutoff = step/100)
        if matches_list:
            break
    # return relevant label from dictionary and get plaintext explanation
    if len(entities_dict) > 0:
        label = entities_dict[matches_list[0]]
        description = spacy.explain(label)
    else:
        label, description = "[---]", "No category found"
    return label, description


if __name__ == "__main__":

    # extract index-title-abstract data from cleaned csv file
    data = []
    with open("cleaned_data/abstracts_over150chars.txt") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            data.append(row)

    # get article data
    sample_index = random.randint(0, len(data))
    for row in data:
        if int(row[0]) == sample_index:
            title = str(row[1])
            abstract = str(row[2])
            print(row[0])
            print(title.upper())
            print(row[2])

    label, description = entity_recognition_fuzzy_match(abstract, title)
    print(label, description)