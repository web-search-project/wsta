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
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    '''
    Slice the dictionary according to key set
    '''
    keys = list(adict.keys())
    dict_slice = {}
    for k in keys[start:end]:
        dict_slice[k] = adict[k]
    return dict_slice

def dict_slice_by_keys(adict, keys):
    '''
    Slice the dictionary according to given key set
    '''
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
    '''
    Generate test csv
    '''
    # Handling test set
    testing_filename = '/content/gdrive/My Drive/Colab Notebooks/test-unlabelled.json'
    testing_set = get_dict_from_file(testing_filename)
    print('test set loaded')
       
    # Documents
    index_file = '/content/gdrive/My Drive/Colab Notebooks/index.json'
    term_dict = get_dict_from_file(index_file)
    wiki_dir = '/content/gdrive/My Drive/Colab Notebooks/wiki-pages-text/'
    print('index loaded')
    
    # Tf-idf
    print('Loading model')
    model_path = '/content/gdrive/My Drive/Colab Notebooks/tfidf.pickle'
    tfidf = load_model(model_path)
    print('tfidf loaded')
   
    # test set key set
    #st_number = 0 
    #slice_number = 1000
    #testing_slice_set = dict_slice(testing_set, st_number, st_number + slice_number)
    testing_slice_set = testing_set
    print('test set slice: %d' %(len(testing_slice_set)))   

    # NER
    ner = NER()
    print('*************************** Start tdidf ***********************')
    process_number = 4
    executor = ThreadPoolExecutor(max_workers=process_number)
    result = []

    t1 = time.time()
    i = 0
    for key, value in testing_slice_set.items():
        query = text_sub(value['claim'])
        query_ner_list = ner.getNER(query)
        result.append(executor.submit(process_test, i, query, query_ner_list))
        i += 1

    dfs = []
    for future in as_completed(result):
        dfs.append(future.result())
    t2 = time.time()
    print('time: %.2fs' %(t2 - t1))

    dataframe = res = pd.concat(dfs, axis=0, ignore_index=True)
    dataframe.to_csv("/content/gdrive/My Drive/Colab Notebooks/test5.csv")
    print(dataframe.shape)

def process_test(i, query, query_ner_list):
    '''
    Each process reads one claim and search related wiki test and store in list
    '''

    key_list = []
    query_list = []
    dr_list = []
    page_list = []
    title_list = []

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
    if i % 100 == 0:
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

if __name__ == '__main__':
    #first_test()
    #sp_test()
    test_generate_csv()
