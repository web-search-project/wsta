import os
import sys
import getopt
import glob as gb
import numpy as np
import multiprocessing
from multiprocessing import Pool, cpu_count, Manager, Lock
import time
from collections import Counter
import re
import gc
import json

def get_self_opt(argv):
    
    global input_dir

    try:
        opts, args = getopt.getopt(argv, "hi:", ["help", "input="])
    except getopt.GetoptError:
        print('Options error!')
        print('Error: check_term.py -i <input_dir>')
        print('   or: check_term.py --input=<input_dir>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('    check_term.py -i <input_dir>')
            print('or: check_term.py --input=<input_dir>')
            sys.exit()
        elif opt in ("-i", "--input"):
            input_dir = arg

def preprocess_file(path):
    print(path + ' starts')
    sharedDict = {}
    dir_name, filename = os.path.split(path)
    file_number = int(re.findall(r"\d{3}", filename)[0])

    with open(path, 'r', encoding='utf-8') as fp:
        cur_term = ''
        term_st_offset = 0
        term_length = 0
        line = fp.readline()
        while line:

            # Add to dict
            terms = line.split()
            term = terms[0]

            if term != cur_term:
                if cur_term != '':
                    sharedDict[cur_term] = [file_number, term_st_offset, term_length]
                cur_term = term
                line_bytes = bytes(line, encoding='utf-8')
                term_length = len(line_bytes)
                term_st_offset = fp.tell() - term_length
            else:
                line_bytes = bytes(line, encoding='utf-8')
                term_length += len(line_bytes)
            line = fp.readline()

        sharedDict[cur_term] = [file_number, term_st_offset, term_length]
        fp.close()
    print(path + ' ends')
    gc.collect()
    return sharedDict

def get_wiki_dict(dir_name, process_number):
    # wiki_text_path store wiki text paths
    input_dir = dir_name
    input_dir = os.path.join(input_dir + '*.txt')
    wiki_text_path = sorted(gb.glob(input_dir))
    print('Number of files: %d' %(len(wiki_text_path)))
    print('Maximum processes: %d' %(process_number))

    result = []
    pool = Pool(processes=process_number)
    for path in wiki_text_path:
        result.append(pool.apply_async(preprocess_file, (path,)))  
    pool.close()
    pool.join()
    term_dict = {}
    for res in result:
        term_dict.update(res.get())
    return term_dict

def get_wiki_dict_from_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as fp:
        term_dict  = json.load(fp)
        fp.close()
    return term_dict

if __name__ == '__main__':

    get_self_opt(sys.argv[1:])
    
    print('------------------------------------------options------------------------------------------')
    print('input_dir: %s' %(input_dir))
    print('---------------------------------------options end-----------------------------------------')
   
    process_number = cpu_count()
    term_dict = get_wiki_dict(input_dir, process_number)
    print('Got dict')

    with open('result.json', 'w') as fp:
        json.dump(term_dict, fp)
        fp.close()
