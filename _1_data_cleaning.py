""" Removes unwanted entries from xml dumps of english wikipedia article abstracts.    
    Extract .gz files from https://dumps.wikimedia.org/enwiki/latest/ and move to "/raw_data" """

import xml.etree.ElementTree as ET
import csv
import os
from unidecode import unidecode


def strip_abstract(title, abstract, strip=False):
    """ removes words found in title from abstract, returns stripped abstract as string """
    if strip == True:
        title = str(title.lower()).split()
        abstract = str(abstract.lower()).split()
        stripped_abstract = [word for word in abstract if word not in title]
        return " ".join(stripped_abstract)


# index count
index = 0

# iterate through files in raw data folder
for file in os.listdir("raw_data"):
    if file.endswith(".xml"):
        current_filepath = os.path.join("raw_data", file)
        print(current_filepath)

        # iterate through all elements in current xml file
        tree = ET.parse(current_filepath)
        for element in tree.iter():   
            if element.tag == "title":
                title = str(element.text[11:])
            
            if element.tag == "abstract":
                # reject if no abstract available
                if element.text == "NoneType":
                    continue
                # reject recurring events eg. "2015 Premiership Rugby Sevens Series" 
                if any(char.isdigit() for char in title):
                    continue
                # reject list articles eg. "List of James Bond films"
                if "list" in title.lower():
                    continue
                
                # separate desired text from data fields
                abstract, sep, tail = str(element.text).partition("|")
                # replace foreign characters with english equivalent
                title, abstract = unidecode(title), unidecode(abstract)
            
                if len(abstract) > 150:
                    # option to remove words found in title from abstract
                    abstract = strip_abstract(title, abstract, strip=True)
                    # save data to csv file
                    with open("cleaned_data/abstracts_over150chars_stripped.txt", "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([str(index), str(title), str(abstract)])
                        index += 1
