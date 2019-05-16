from check_term import get_wiki_dict
from check_term import get_wiki_dict_from_file
import json
import glob as gb
import os
'''
term_dict = get_wiki_dict("wiki-pages-text/", 8)
print(len(term_dict))

with open('result.json', 'w') as fp:
    json.dump(term_dict, fp)
    fp.close()
'''
term_dict_2 = get_wiki_dict_from_file('result.json')
print(len(term_dict_2))

file_number, start_pos, read_len = term_dict_2['World_record_progression_4_×_100_metres_freestyle_relay']
#file_number, start_pos, read_len = term_dict_2['Wellington_Sandoval']
print('%d: %d: %d' %(file_number, start_pos, read_len))

wiki_text_path = sorted(gb.glob('wiki-pages-text/*.txt'))
path = wiki_text_path[file_number - 1]
dir_name, filename = os.path.split(path)
print(filename)
with open(path, 'br') as fp:
    fp.seek(start_pos)
    term_byte = fp.read(read_len)
    term_str = str(term_byte, encoding='utf-8')
    print(term_str)
    fp.close()
