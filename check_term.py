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

class IndexValue:
    def __init__(self, ifile, st_offset, length):
        self.ifile = ifile
        self.st_offset = st_offset
        self.length = length

    def __str__(self):
        return 'File number: ' + self.ifile + ' start offset: ' + self.st_offset + ' length: ' + self.length

    def get_value(self):
        return self.ifile, self.st_offset, self.length

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

def preprocess_file(path, sharedDict, lock):
    print(path + ' starts')
    dir, filename = os.path.split(path)
    file_number = re.findall(r"\d{3}", filename)[0]

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
                    with lock:
                        sharedDict[term] = IndexValue(file_number, term_st_offset, term_length)
                cur_term = term
                term_length = len(line)
                term_st_offset = fp.tell() - term_length
            else:
                term_length += len(line)

            line = fp.readline()
        fp.close()
    print(path + ' ends')

if __name__ == '__main__':

    get_self_opt(sys.argv[1:])
    
    print('------------------------------------------options------------------------------------------')
    print('input_dir: %s' %(input_dir))
    print('---------------------------------------options end-----------------------------------------')
    
    # wiki_text_path store wiki text paths
    input_dir = os.path.join(input_dir + '*.txt')
    wiki_text_path = sorted(gb.glob(input_dir))
    print('Number of files: %d' %(len(wiki_text_path)))

    process_number = int(cpu_count())
    print('Maximum processes: %d' %(process_number))

    manager = Manager()
    term_dict = manager.dict()
    lock=manager.Lock()
    pool = Pool(processes=process_number)
    for path in wiki_text_path:
        pool.apply_async(preprocess_file, (path, term_dict, lock))    
    pool.close()
    pool.join()

    print('Got dict')

    with open('result.txt', 'w') as fp:
        for key, value in term_dict.items():
            fp.write(key + '\n')
            fp.write('\t' + value + '\n')
        fp.close()

