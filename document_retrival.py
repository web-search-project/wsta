from docutil import DocumentRetrival
from docutil import get_dict_from_file
from docutil import save_dict

from ner import NER

from mytfidf import Tfidf
from mytfidf import text_sub
from mytfidf import save_model
from mytfidf import load_model

import nltk

def dict_slice(adict, start, end):
    keys = list(adict.keys())
    dict_slice = {}
    for k in keys[start:end]:
        dict_slice[k] = adict[k]
    return dict_slice

def print_dict(adict):
    for key, value in adict.items():
        print('%s : %s' %(key, str(value)))

def first_test():
    # Download nltk
    nltk.download('punkt')

    # Handling training set
    training_filename = '../data/train.json'
    training_set = get_dict_from_file(training_filename)
    training_set = dict_slice(training_set, 0, 10)
    print_dict(training_set)

    evidence_docs = {}
    for key, value in training_set.items():
        doc_set = set()
        for doc, page in value['evidence']:
            doc_set.add(doc)
        evidence_docs[key] = doc_set
    print(len(evidence_docs))
    print_dict(evidence_docs)

    # Documents
    index_file = './index.json'
    dr = DocumentRetrival('../wiki-pages-text/')
    term_dict = dr.get_term_dict(index_file)
    doc_list = list(term_dict.keys())

    # Tf-idf
    print('Loading model')
    model_path = 'tfidf.pickle'
    tfidf = load_model(model_path)
    if tfidf is None:
        tfidf = Tfidf(doc_ids=doc_list)
        ts = tfidf.fit_transform()
        print('Save model')
        print(tfidf.matrix)
        save_model(tfidf, model_path)
    
    # NER
    ner = NER()
    print('*************************** Start tdidf ***********************')
    for key, value in training_set.items():
        query = text_sub(value['claim']) 
        query_ner_list = ner.getNER(query)
        if len(query_ner_list) == 0:
            query_ner_list = [query]
        print("NER: %s" %(str(query_ner_list)))
        res = tfidf.cos_similarity(query_ner_list)
        print(res)
        print(evidence_docs[key])
        print('')

if __name__ == '__main__':
    first_test()
