import nltk
import string
import os
import re
import pickle
import time
from math import log, sqrt
from collections import Counter

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.util import ngrams

nltk.download('punkt')
nltk.download('stopwords')
stop_words = stopwords.words('english')

def save_model(model, model_path):
    with open(model_path, 'wb') as fp:
        pickle.dump(model, fp)
        
def load_model(model_path):
    if os.path.exists(model_path):
        with open(model_path, 'rb') as fp:
            return pickle.load(fp)
    return None

def tokenize(text):
    # Remove unwanted structure
    text = re.sub('_', ' ', text)
    text = re.sub('-LRB-', ' ', text)
    text = re.sub('-RRB-', ' ', text)
    first_tokens = nltk.word_tokenize(text.lower())
    first_tokens = [word for word in first_tokens if word not in stop_words]
    first_tokens = [word for word in first_tokens if word not in string.punctuation]
    first_tokens = [PorterStemmer().stem(word) for word in first_tokens]
    return first_tokens

def vbyte_encode(num):

    # out_bytes stores a list of output bytes encoding the number
    out_bytes = []
    
    while num >= 128:
        out_bytes.append(bytes([num % 128]))
        num = int(num / 128)
    out_bytes.append(bytes([num + 128]))
    
    return out_bytes

def vbyte_decode(input_bytes, idx):
    
    # x stores the decoded number
    x = 0
    # consumed stores the number of bytes consumed to decode the number
    consumed = 0

    #input_byte store the integer converted from bytes
    input_byte = int.from_bytes(input_bytes[idx], byteorder = 'big')
    while(input_byte < 128):
        x = x ^ (input_byte << consumed)
        consumed += 7
        idx += 1
        input_byte = int.from_bytes(input_bytes[idx], byteorder = 'big')
    x = x ^ ((input_byte - 128) << consumed)
    consumed = int(consumed / 7)
    consumed += 1
    
    return x, consumed

def decompress_list(input_bytes, gapped_encoded):
    res = []
    prev = 0
    idx = 0
    while idx < len(input_bytes):
        dec_num, consumed_bytes = vbyte_decode(input_bytes, idx)
        idx += consumed_bytes
        num = dec_num + prev
        res.append(num)
        if gapped_encoded:
            prev = num
    return res

def fit(raw_docs, ngram=(1,1)):

    # processed_docs stores the list of processed docs
    processed_docs = []
    # vocab contains (term, term id) pairs
    vocab = {}
    # total_tokens stores the total number of tokens
    total_tokens = 0

    min_ngram, max_ngram = ngram

    # doc_term_freqs stores the counters (mapping terms to term frequencies) of all documents
    doc_term_freqs = []

    for raw_doc in raw_docs:
    
        # norm_doc stores the normalized tokens of a doc
        norm_doc = tokenize(raw_doc)
        processed_docs.append(norm_doc)
        
        doc_term_cnt = Counter()
        for n in range(min_ngram, max_ngram + 1):
            cur_ngram = list(ngrams(norm_doc, n))
            doc_term_cnt.update(cur_ngram)
            for word in cur_ngram:
                if word not in vocab:
                    vocab[word] = len(vocab)
                total_tokens += 1
        doc_term_freqs.append(doc_term_cnt)
        #if len(processed_docs) % 100000 == 0:
            #print(len(processed_docs))
    
    return processed_docs, vocab, total_tokens, doc_term_freqs

class InvertedIndex:
    def __init__(self, vocab, doc_term_freqs):
        self.vocab = vocab
        self.doc_len = [0] * len(doc_term_freqs)
        self.doc_term_freqs = [[] for i in range(len(vocab))]
        self.doc_ids = [[] for i in range(len(vocab))]
        self.doc_freqs = [0] * len(vocab)
        self.total_num_docs = 0
        self.max_doc_len = 0
        for docid, term_freqs in enumerate(doc_term_freqs):
            doc_len = sum(term_freqs.values())
            self.max_doc_len = max(doc_len, self.max_doc_len)
            self.doc_len[docid] = doc_len
            self.total_num_docs += 1
            for term, freq in term_freqs.items():
                term_id = vocab[term]
                self.doc_ids[term_id].append(docid)
                self.doc_term_freqs[term_id].append(freq)
                self.doc_freqs[term_id] += 1

        for freqs in range(len(self.doc_term_freqs)):
            
            #freq_bytes stores the encoded bytes of frequency
            freq_bytes = []
            
            for i in range(len(self.doc_term_freqs[freqs])):
                freq_bytes.extend(vbyte_encode(self.doc_term_freqs[freqs][i]))
            self.doc_term_freqs[freqs] = freq_bytes
                
        for ids in range(len(self.doc_ids)):
            
            #prev stores the previous value
            prev = 0
            
            #doc_bytes stores the encoded bytes of document id
            doc_bytes = []
            
            for i in range(len(self.doc_ids[ids])):
                doc_bytes.extend(vbyte_encode(self.doc_ids[ids][i] - prev))
                prev = self.doc_ids[ids][i]
            self.doc_ids[ids] = doc_bytes
            
    def num_terms(self):
        return len(self.doc_ids)

    def num_docs(self):
        return self.total_num_docs

    def docids(self, term):
        term_id = self.vocab[term]
        # We decompress
        return decompress_list(self.doc_ids[term_id], True)

    def freqs(self, term):
        term_id = self.vocab[term]
        # We decompress
        return decompress_list(self.doc_term_freqs[term_id], False)

    def f_t(self, term):
        term_id = self.vocab[term]
        return self.doc_freqs[term_id]

    def space_in_bytes(self):
        # this function assumes the integers are now bytes
        space_usage = 0
        for doc_list in self.doc_ids:
            space_usage += len(doc_list)
        for freq_list in self.doc_term_freqs:
            space_usage += len(freq_list)
        return space_usage

def query_tfidf(query, index, ngram=(1, 1), k=3):

    # scores stores doc ids and their scores
    scores = Counter()

    # number of documents
    n = index.num_docs()
    print('n: %d' %(n))

    min_ngram, max_ngram = ngram
    query_list = []
    for i in range(min_ngram, max_ngram + 1):
        query_list.extend(list(ngrams(query, i)))

    for query_term in query_list:
        # count the file that include the term
        count = 0
        for doc_id in index.docids(query_term):
            #tf stores the calculation result of tf
            tf = log(1 + index.freqs(query_term)[count])
            #idf stores the calculation result of idf
            idf = log(n / index.f_t(query_term))
            #score stores score of each term in each file
            score = (1 / sqrt(index.doc_len[doc_id])) * tf * idf
            scores.update({doc_id: score})
            count += 1

    return scores.most_common(k)

def test():
    import json
    index = {}
    with open('index.json', 'r') as fp:
        index = json.load(fp)
    raw_docs = list(index.keys())[:1000000]

    print('Start processing documents')
    processed_docs, vocab, total_tokens, doc_term_freqs = fit(raw_docs, ngram = (1, 1))

    print("Number of documents = {}".format(len(processed_docs)))
    #print(processed_docs)
    print("Number of unique terms = {}".format(len(vocab)))
    #print(vocab)
    print("Number of tokens = {}".format(total_tokens))

    print('doc_term_freqs: %d' %(len(doc_term_freqs)))
    print(doc_term_freqs[1])


    invindex = InvertedIndex(vocab, doc_term_freqs)

    # print inverted index stats
    print("documents = {}".format(invindex.num_docs()))
    print("number of terms = {}".format(invindex.num_terms()))
    print("longest document length = {}".format(invindex.max_doc_len))
    print("uncompressed space usage MiB = {:.3f}".format(invindex.space_in_bytes() / (1024.0 * 1024.0)))

    query = "1986 NBA Finals"
    stemmed_query = tokenize(query)
    t1 = time.time()
    results = query_tfidf(stemmed_query, invindex, ngram=(1,1))
    t2 = time.time()
    print('Each query time: %.2f' %(t2 - t1))
    for rank, res in enumerate(results):
        print("RANK {:2d} DOCID {:8d} SCORE {:.3f} CONTENT {:}".format(rank+1,res[0],res[1],raw_docs[res[0]]))


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

if __name__ == '__main__':
    test()
