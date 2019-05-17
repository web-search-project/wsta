import nltk
import string
import os
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

def save_model(model, model_path):
    with open(model_path, 'wb') as fp:
        pickle.dump(model, fp)
        
def load_model(model_path):
    if os.path.exists(model_path):
        with open(model_path, 'rb') as fp:
            return pickle.load(fp)
    return None

def text_sub(text):
    text = re.sub('_', ' ', text)
    text = re.sub('-LRB-', '(', text)
    text = re.sub('-RRB-', ')', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def tokenize(text):
    text = text_sub(text)
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

class Tfidf:
    def __init__(self, matrix={}, doc_ids=[]):
        self._matrix = matrix
        self._tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', ngram_range=(1, 1))
        self.doc_ids = doc_ids

    @property
    def matrix(self):
        return self._matrix

    @property
    def tfidf(self):
        return self._tfidf

    def fit_transform(self):
        self._matrix = self._tfidf.fit_transform(self.doc_ids)
        return self._matrix

    def transform(self, str_list):
        return self._tfidf.transform(str_list)

    def cos_similarity(self, query_list, number=3):
        result = {}
        query_tfidf = self._tfidf.transform(query_list)
        for i, query in enumerate(query_tfidf):
            cosine_similarities = cosine_similarity(query, self._matrix).flatten()
            related_docs_indices = cosine_similarities.argsort()[:(-1 - number):-1]
            cur_doc_list = []
            for doc_id in related_docs_indices:
                cur_doc_list.append(self.doc_ids[doc_id])
            result[query_list[i]] = cur_doc_list
        return result 

