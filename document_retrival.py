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
    #print_dict(training_set)

    evidence_docs = {}
    for key, value in training_set.items():
        doc_set = set()
        for doc, page in value['evidence']:
            doc_set.add(doc)
        evidence_docs[key] = doc_set
    #print(len(evidence_docs))
    #print_dict(evidence_docs)

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
        save_model(tfidf, model_path)
    
    # NER
    ner = NER()
    print('*************************** Start tdidf ***********************')
    res_dict = {}
    for key, value in training_set.items():
        query = text_sub(value['claim']) 
        query_ner_list = ner.getNER(query)
        if len(query_ner_list) == 0:
            query_ner_list = [query]
        #print("NER: %s" %(str(query_ner_list)))
        res = tfidf.cos_similarity(query_ner_list, number = 5)

        for sub_query in res.keys():
            sub_query_list = res[sub_query]
            for sub_query_doc in sub_query_list:
                if sub_query == text_sub(sub_query_doc):
                    sub_query_match = [sub_query_doc]
                    res[sub_query] = sub_query_match
                    break

        res_dict[key] = res
        print(res)
        print(evidence_docs[key])
        print('')

    # Evaluation
    precision_total = 0.0
    recall_total = 0.0
    TP = 0
    for key in res_dict.keys():
        query_res_dict = res_dict[key]
        doc_ground_true = evidence_docs[key]
        recall_total += len(doc_ground_true)
        doc_rt = []
        for query_res_list in query_res_dict.values():
            doc_rt.extend(query_res_list)
        doc_rt = set(doc_rt)
        precision_total += len(doc_rt)
        for doc in doc_ground_true:
            if doc in doc_rt:
                TP += 1
    print('TP: %d' %(TP))
    print('Precision total: %d' %(precision_total))
    print('Recall total: %d' %(recall_total))
    print('Precision: %.2f' %(TP / precision_total))
    print('Recall: %.2f' %(TP / recall_total))

if __name__ == '__main__':
    first_test()
