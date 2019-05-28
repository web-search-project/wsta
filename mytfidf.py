import nltk
import string
import os
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

#nltk.download('punkt')
#nltk.download('stopwords')
stop_words = stopwords.words('english')

def save_model(model, model_path):
    '''
    Use pickle to save the model
    '''
    with open(model_path, 'wb') as fp:
        pickle.dump(model, fp)
        
def load_model(model_path):
    '''
    Load model
    '''
    if os.path.exists(model_path):
        with open(model_path, 'rb') as fp:
            return pickle.load(fp)
    return None

def text_sub(text):
    '''
    preprocess the sentence, remove punctuation
    '''
    text = re.sub('_', ' ', text)
    text = re.sub('-LRB-', ' ', text)
    text = re.sub('-RRB-', ' ', text)
    text = re.sub('-LSB-', ' ', text)
    text = re.sub('-RSB-', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def tokenize(text):
    '''
    Method to tokenize
    '''
    # Remove unwanted structure
    text = re.sub('_', ' ', text)
    text = re.sub('-LRB-', ' ', text)
    text = re.sub('-RRB-', ' ', text)
    first_tokens = nltk.word_tokenize(text.lower())
    first_tokens = [word for word in first_tokens if word not in stop_words]
    first_tokens = [word for word in first_tokens if word not in string.punctuation]
    first_tokens = [PorterStemmer().stem(word) for word in first_tokens]
    return first_tokens

class Tfidf:
    def __init__(self, matrix={}, doc_ids=[]):
        self._matrix = matrix
        # Build a tfidf model
        self._tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', ngram_range=(1, 2))
        self.doc_ids = doc_ids

    @property
    def matrix(self):
        return self._matrix

    @property
    def tfidf(self):
        return self._tfidf

    def fit_transform(self):
        '''
        Read documents to train models
        '''
        self._matrix = self._tfidf.fit_transform(self.doc_ids)
        return self._matrix

    def transform(self, str_list):
        '''
        Transform query to vector
        '''
        return self._tfidf.transform(str_list)

    def cos_similarity(self, query_list, number=3):
        '''
        Calculate cosine similarity between query and documents
        And return the maximum cosine similarity documents
        '''
        result = {}
        query_tfidf = self._tfidf.transform(query_list)
        for i, query in enumerate(query_tfidf):
            cosine_similarities = cosine_similarity(query, self._matrix).flatten()
            '''
            # make pairs of (index, similarity)
            cosine_similarities = list(enumerate(cosine_similarities))
            # get the tuple with max similarity
            most_similar, similarity = max(cosine_similarities, key=lambda t:t[1])
            cur_doc_list = [self.doc_ids[most_similar]]
            '''
            related_docs_indices = cosine_similarities.argsort()[:(-1 - number):-1]
            cur_doc_list = []
            for doc_id in related_docs_indices:
                cur_doc_list.append(self.doc_ids[doc_id])
            result[query_list[i]] = cur_doc_list
        return result 

