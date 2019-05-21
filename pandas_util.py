import pandas as pd
import json
from docutil import get_dict_from_file
from docutil import DocumentRetrival
from document_retrival import dict_slice

label_dict = {"REFUTES" : 0, "SUPPORTS" : 1, "NOT ENOUGH INFO" : -1}

def read_json(filename, save_csv=False, csv_filename=None):
    term_dict = get_dict_from_file("./index.json")
    dr = DocumentRetrival("../wiki-pages-text/", term_dict)
    term_columns = ['claim', 'evidence', 'label']
    df = pd.DataFrame(columns=term_columns)
    train_dict = get_dict_from_file(filename)
    train_dict = dict_slice(train_dict, 0, 1000)
    for key, value in train_dict.items():
        claim = value['claim']
        label = value['label']
        for term, page in value['evidence']:
            try:
                evidence_read = dr.doc_retrive(term)
            except:
                pass
            #print(evidence_read)
            evidence_read_list = evidence_read.splitlines()
            #print(evidence_read_list)
            for line in evidence_read_list:
                #print(line)
                tokens = line.split()
                if int(tokens[1]) == int(page):
                    #print(tokens[2])
                    df.loc[df.shape[0]+1] = {'claim' : claim, 'evidence' : " ".join(tokens[2:]), 'label': label_dict[label]}
                    break
    if save_csv:
        df.to_csv(csv_filename)

    return df


df = read_json("../data/train.json", save_csv=True, csv_filename='train.csv')
print(df.shape)
