import json
from docutil import get_dict_from_file

training_set = get_dict_from_file('../data/train.json')
max_length = 0
for key, value in training_set.items():
    claim = value['claim']
    claim_word_list = claim.split(' ')
    if(len(claim_word_list)) > max_length:
        max_length = len(claim_word_list)
print(max_length)
