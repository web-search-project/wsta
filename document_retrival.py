from docutil import get_dict_from_file
from docutil import save_dict
from docutil import doc_retrive
from docutil import DocumentRetrival

from ner import NER

from mytfidf import Tfidf
from mytfidf import text_sub
from mytfidf import save_model
from mytfidf import load_model

from multiprocessing import Pool, cpu_count 

import nltk
import time
import json
import math
import gc
import pandas as pd

#nltk.download('punkt')
#nltk.download('stopwords')

''' Constant in MPI '''
#comm = MPI.COMM_WORLD
#rank = comm.rank
#size = comm.size


def dict_slice(adict, start, end):
    keys = list(adict.keys())
    dict_slice = {}
    for k in keys[start:end]:
        dict_slice[k] = adict[k]
    return dict_slice

def dict_slice_by_keys(adict, keys):
    dict_slice = {}
    for k in keys:
        dict_slice[k] = adict[k]
    return dict_slice

def print_dict(adict):
    for key, value in adict.items():
        print('%s : %s' %(key, str(value)))

def sp_test():
    if rank == 0:
        testing_filename = '../data/test-unlabelled.json'
        testing_set = get_dict_from_file(testing_filename)
        testing_set = dict_slice(testing_set, 0, 100)
        data = list(testing_set.keys())
        num_each_process = math.ceil(float(len(testing_set)) / size)
        print(num_each_process)
       
        res = {} 
        for i in range(size):
            st_test = int(i * num_each_process)
            ed_test = min(len(testing_set), int(st_test + num_each_process))
            testing_key_set = data[st_test : ed_test]
            res[str(i)] = testing_key_set
            print('%d finish' %(i))
        save_dict(res, '../data/test-key.json')

def test_generate_csv():
    # Handling test set
    testing_filename = '../data/test-unlabelled.json'
    testing_set = get_dict_from_file(testing_filename)
    testing_set = dict_slice(testing_set, 0, 100)
    print('test set loaded')
    
    
    # Documents
    index_file = './index.json'
    term_dict = get_dict_from_file(index_file)
    wiki_dir = '../wiki-pages-text/'
    print('index loaded')
    
    # Tf-idf
    print('Loading model')
    model_path = './tfidf.pickle'
    tfidf = load_model(model_path)
    print('tfidf loaded')
   
    # test set key set
    slice_number = 100
    testing_slice_set = dict_slice(testing_set, 0, slice_number)
    print('test set slice: %d' %(slice_number))
    
    
    # NER
    ner = NER()
    print('*************************** Start tdidf ***********************')
    process_number = 2
    result = []
    pool = Pool(processes=process_number)
    i = 0
    for key, value in testing_slice_set.items():
        result.append(pool.apply_async(process_test, (i, key, value, ner, tfidf, term_dict, wiki_dir)))  
        i += 1
    pool.close()
    pool.join()
    t_dict = {}

    df_list = [res.get() for res in result]
        
    dataframe = res = pd.concat(df_list, axis=0, ignore_index=True)
    dataframe.to_csv("../data/test.csv")
    print(dataframe.shape)

def process_test(i, key, value, ner, tfidf, term_dict, wiki_dir):
    key_list = []
    query_list = []
    dr_list = []
    page_list = []
    title_list = []
    query = text_sub(value['claim']) 
    query_ner_list = ner.getNER(query)
    res = tfidf.cos_similarity(query_ner_list, number = 2)

    for sub_query in res.keys():
        sub_query_list = res[sub_query]
        for sub_query_doc in sub_query_list:
            doc_read = doc_retrive(sub_query_doc, term_dict, wiki_dir)
            doc_read_list = doc_read.splitlines()
            for doc_line in doc_read_list:
                tokens = doc_line.split()
                key_list.append(key)
                query_list.append(query)
                title_list.append(tokens[0])
                page_list.append(tokens[1])
                dr_list.append(" ".join(tokens[2:]))

    df = pd.DataFrame({'key' : key_list, 'claim' : query_list, 'evidence' : dr_list, 'title' : title_list, 'page' : page_list})
    gc.collect()
    if i % 10 == 0:
        print(i)
    return df

def save_csv():
    if rank == 0:
        df_dict = temp_df
        for df in dfs:
            for key in df.keys():
                df_dict[key].extend(df[key])
        dataframe = pd.DataFrame(df_dict)
        dataframe.to_csv("/home/yfedward/WT/wsta/data/test.csv")
        print(dataframe.shape)

def first_test():

    # Handling training set
    training_filename = '../data/train.json'
    training_set = get_dict_from_file(training_filename)
    training_set = dict_slice(training_set, 0, 100)
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
        #print("NER: %s" %(str(query_ner_list)))
        res = tfidf.cos_similarity(query_ner_list, number = 2)

        for sub_query in res.keys():
            sub_query_list = res[sub_query]
            for sub_query_doc in sub_query_list:
                if sub_query == text_sub(sub_query_doc):
                    sub_query_match = [sub_query_doc]
                    res[sub_query] = sub_query_match
                    break

        res_dict[key] = res
        #print(res)
        #print(evidence_docs[key])
        #print('')

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
    #first_test()
    #sp_test()
    test_generate_csv()
