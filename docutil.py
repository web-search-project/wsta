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

def get_dict_from_file(file_name):
    '''
    Load json file to a dict.
    '''
    with open(file_name, 'r', encoding='utf-8') as fp:
        term_dict  = json.load(fp)
        return term_dict


def save_dict(term_dict, file_name):
    '''
    Save a json dict to json file
    '''
    with open(file_name, 'w') as fp:
        json.dump(term_dict, fp)
        fp.close()

class DocumentRetrival:
    '''
    This class is used to retrive documents from wiki texts
    '''

    def __init__(self, wiki_dir, term_dict=None):
        # wiki_dir is a string that indicates the path
        self.wiki_dir = wiki_dir
        # term_dict is a index dict that indicates the information of a term in wiki
        self.term_dict = term_dict

    def is_wiki_dir_exist(self):
        return os.path.exists(self.wiki_dir)

    def preprocess_file(self, path):
        '''
        Read a wiki text and get the file number, offset and the length of term information

        '''
        # sharedDict is an index dict
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
        print(path + ' finished')
        gc.collect()
        return sharedDict

    def get_wiki_dict(self, process_number):
        '''
        This method uses multiprocessing to read each wiki text and build an index of terms of wiki
        '''
        # wiki_text_path store wiki text paths
        input_dir = self.wiki_dir
        input_dir = os.path.join(input_dir + '*.txt')
        wiki_text_path = sorted(gb.glob(input_dir))
        print('Number of files: %d' %(len(wiki_text_path)))
        print('Maximum processes: %d' %(process_number))

        result = []
        pool = Pool(processes=process_number)
        for path in wiki_text_path:
            result.append(pool.apply_async(self.preprocess_file, (path,)))  
        pool.close()
        pool.join()
        t_dict = {}
        for res in result:
            t_dict.update(res.get())
        return t_dict
 
    def get_term_dict(self, file_name):
        '''
        Get the dict of index with the file_name, If the file exists, read the file.
        If not, generate index by get_wiki_dict method
        '''
        if self.term_dict != None:
            return self.term_dict
        if os.path.exists(file_name):
            try:
                term_dict = get_dict_from_file(file_name)
                assert(len(term_dict) > 5000000)
                self.term_dict = term_dict
                return term_dict
            except:
                pass

        if not self.is_wiki_dir_exist():
            return None
        
        # The number of process
        process_number = int(cpu_count() / 2 + 1)
        term_dict = self.get_wiki_dict(process_number)
        self.term_dict = term_dict
        save_dict(self.term_dict, file_name)
        return term_dict
  
    def doc_retrive(self, term):
        '''
        According to the term given, get the information of the term in the wiki text
        '''
        if self.term_dict is not None:
            file_number, start_pos, read_len = self.term_dict[term]
            wiki_text_path = sorted(gb.glob(self.wiki_dir + '*.txt'))
            path = wiki_text_path[file_number - 1]
            dir_name, filename = os.path.split(path)
            with open(path, 'br') as fp:
                fp.seek(start_pos)
                term_byte = fp.read(read_len)
                term_str = str(term_byte, encoding='utf-8')
                fp.close()
            return term_str
        return None

def doc_retrive(term, term_dict, wiki_dir):
    '''
    Another method to get information of term
    Argument:
        term_dict: The index file
        wiki_dir: wiki-text directory path
    
    '''
    if term_dict is not None:
        file_number, start_pos, read_len = term_dict[term]
        wiki_text_path = sorted(gb.glob(wiki_dir + '*.txt'))
        path = wiki_text_path[file_number - 1]
        dir_name, filename = os.path.split(path)
        with open(path, 'br') as fp:
            fp.seek(start_pos)
            term_byte = fp.read(read_len)
            term_str = str(term_byte, encoding='utf-8')
            fp.close()
        return term_str
    return None

def main():
    args = sys.argv[1:]
    if len(args) != 1:
        print('Error: please indicate index json file name.')
        exit()
    index_file = args[0]

    dr = DocumentRetrival('../wiki-pages-text/')
    
    term_dict = dr.get_term_dict(index_file)
    print('Got dict')
    print(len(term_dict))

    term_str = dr.doc_retrive('World_record')
    print(term_str)

if __name__ == '__main__':
    main()
    
