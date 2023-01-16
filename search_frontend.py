import csv
import gzip
import math
import operator
import os
import pickle
import re
from collections import Counter, defaultdict
import  flask
import numpy
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import inverted_index_title_gcp
import nltk
from nltk.corpus import stopwords
import threading
TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer
from contextlib import closing


def get_csv():
    pr_dict = {}
    with gzip.open("pr/part-00000-be7b3754-1360-4de9-92ff-52da9c181d30-c000.csv.gz", "rt") as csvFile:
        csvreader = csv.reader(csvFile)
        for row in csvreader:
            pr_dict.update({int(row[0]): float(row[1])})
    return pr_dict


def get_pv():
    with open("pv_clean",'rb') as f:
        wid2pv=pickle.load(f)
    return  wid2pv


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        self.N=6348910
        self.k1=1.5
        self.b =0.75

        self.title_index_path = "postings_gcp_title"
        self.index_title = inverted_index_gcp.InvertedIndex.read_index(self.title_index_path, 'index_title')

        self.anchor_index_path="postings_gcp_anchor"
        self.index_anchor=inverted_index_gcp.InvertedIndex.read_index(self.anchor_index_path, 'index_anchor')

        self.text_index_path = "postings_gcp_text"
        self.index_text = inverted_index_gcp.InvertedIndex.read_index(self.text_index_path, 'index_text')

        self.flag=False
        self.pr_dict=get_csv()
        
        self.pv_dict=get_pv()

        self.DL={}


        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)






app = MyFlaskApp(__name__)
nltk.download('stopwords')
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


def combine_dl(index):
    for doc_id, dl in index.DL.items():
        if doc_id in app.index_title.id_title_dict:
            app.DL[doc_id]+=dl
combine_dl(app.index_text)
combine_dl(app.index_title)
combine_dl(app.index_anchor)



@app.route("/search")
def search():


    app.flag=True
    res = []
    query = request.args.get('query', '')
    x = 90
    flag_question= False
    if query[-1] == '?' or query[0] in ["What","Which","Who","Where","Why","When","How","Whose"]:
        x = 1350
        flag_question=True
    y=8
    if len(query) <3 :
        y=30
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query = tokenization_stopwords(query)
    counter_query = Counter(query)

    text_data = search_bm25()
    title_data = search_title()
    title_data = title_data
    anchor_data = search_anchor()
    anchor_data = anchor_data
    ranked = Counter()
    i=0
    k=0
    j=0
    for doc_id, title  in text_data :
        ranked[doc_id]+= x/(i+1)
        i+=1


    for doc_id, title in title_data:
        ranked[doc_id] += y


    for doc_id, title in anchor_data:
        ranked[doc_id] += 4.5



    for doc_id , score in ranked.items() :
        if app.pr_dict[doc_id]>1:
            pr=int(math.log10(app.pr_dict[doc_id]))
        else:
            pr = app.pr_dict[doc_id]
        if app.pv_dict[doc_id] > 1 &  len(query) < 3:
            pv=int(math.log2(app.pv_dict[doc_id]))
        else:
            pv=app.pv_dict[doc_id]
        ranked[doc_id]=ranked[doc_id] * pr*pv

    res = [(str(k),app.index_title.id_title_dict[k]) for k, v in sorted(ranked.items(), key=lambda item: item[1], reverse=True)][:100]

    # END SOLUTION
    app.flag = False
    return jsonify(res)



@app.route("/search_bm25")
def search_bm25():
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    AVGDL = sum(app.index_text.DL.values()) / app.N
    candidates=Counter()
    query = tokenization_stopwords(query)
    for term in query: #run on each term in the query
        if term in app.index_text.df: #check that the word is in the corpus
            posting_list = read_posting_list(app.index_text, term,"postings_gcp_text/") #get the pls of the word -> {(doc_id,f)...}
            df = app.index_text.df[term]#number of documnets that  contains the 'term'
            idf = math.log(1 + (app.N - df + 0.5) / (df + 0.5))
            for doc_id, freq in posting_list:#run on the pls of the term
                if doc_id in app.index_text.DL.keys():#if the doc_id exist in the corpus
                    len_doc=app.index_text.DL[doc_id]
                    numerator=idf*freq*(app.k1+1)
                    denominator = freq + app.k1 * (1 - app.b + app.b * len_doc / AVGDL)
                    bm25_score=numerator/denominator
                    candidates[doc_id] += bm25_score

    res = [(k,app.index_title.id_title_dict[k]) for k, v in sorted(candidates.items(), key=lambda item: item[1], reverse=True)][:100]
    # END SOLUTION
    if app.flag==True:
        return res
    return jsonify(res)





@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query = tokenization_stopwords(query)
    counter_query=Counter(query)
    similarities = defaultdict(int)
    for term in query:
        if term in app.index_text.df:
            idf=math.log2(len(app.index_text.DL) / app.index_text.df[term])
            posting_list= read_posting_list(app.index_text, term,"postings_gcp_text/")
            for doc_id,freq in posting_list:
                tf=freq/app.index_text.DL[doc_id]
                weight=tf*idf
                similarities[doc_id] += weight*counter_query[term]


    # normalization for the query = 1/|q|
    normalization_query = 0
    sum_q=0
    for term,freq in counter_query.items():
        sum_q += freq*freq
    normalization_query=1/math.sqrt(sum_q)

    for doc_id in similarities.keys():
            nf = 1/math.sqrt(app.index_text.nf[doc_id])
            similarities[doc_id] *= normalization_query*nf
    res = [(k, app.index_title.id_title_dict[k]) for k, v in
           sorted(similarities.items(), key=lambda item: item[1], reverse=True)][:100]
    # END SOLUTION
    if app.flag == True:
        return res
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    query = tokenization_stopwords(query)
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    candidates = {}
    for word in np.unique(query):
        if word in app.index_title.df:
            with closing(inverted_index_gcp.MultiFileReader()) as reader:
                locs = app.index_title.posting_locs[word]
                n_byte_array = app.index_title.df[word] * TUPLE_SIZE
                b = reader.read(locs, n_byte_array,"postings_gcp_title/")
                for i in range(app.index_title.df[word]):
                    doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                    if doc_id not in candidates:
                        candidates[doc_id] = 1
                    else:
                        candidates[doc_id] += 1

    res = [(k, app.index_title.id_title_dict[k]) for k, v in
           sorted(candidates.items(), key=lambda item: item[1], reverse=True)]
    if app.flag == True:
        return res
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).
        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    query = tokenization_stopwords(query)
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    candidates = {}
    for word in np.unique(query):
        if word in app.index_anchor.df:
            with closing(inverted_index_gcp.MultiFileReader()) as reader:
                locs = app.index_anchor.posting_locs[word]
                n_byte_array = app.index_anchor.df[word] * TUPLE_SIZE
                b = reader.read(locs, n_byte_array, "postings_gcp_anchor/")
                for i in range(app.index_anchor.df[word]):
                    doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                    if doc_id not in candidates:
                        candidates[doc_id] = 1
                    else:
                        candidates[doc_id] += 1

    res = [(k, app.index_title.id_title_dict[k]) for k, v in
           sorted(candidates.items(), key=lambda item: item[1], reverse=True) if k in app.index_title.id_title_dict]

    if app.flag == True:
        return res
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''

    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    for doc_id in wiki_ids:
        try:
            res.append(app.pr_dict[doc_id])
        except:
            continue
    # END SOLUTION
    return jsonify(res)





@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    for doc_id in wiki_ids:
        try:
            res.append(app.pv_dict[doc_id])
        except:
            continue
    # END SOLUTION
    return jsonify(res)




def read_posting_list(inverted, w,base_dir=""):
    with closing(inverted_index_gcp.MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE,base_dir)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list


def tokenization_stopwords(data):
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb']
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    all_stopwords = english_stopwords.union(corpus_stopwords)
    tokens = [token.group() for token in RE_WORD.finditer(data.lower())]
    tokens = [token for token in tokens if token not in all_stopwords]
    return tokens


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
