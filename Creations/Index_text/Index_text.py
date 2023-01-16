#!/usr/bin/env python
# coding: utf-8

# ***Important*** DO NOT CLEAR THE OUTPUT OF THIS NOTEBOOK AFTER EXECUTION!!!

# In[1]:


# if the following command generates an error, you probably didn't enable 
# the cluster security option "Allow API access to all Google Cloud services"
# under Manage Security → Project Access when setting up the cluster
get_ipython().system('gcloud dataproc clusters list --region us-central1')


# # Imports & Setup

# In[2]:


get_ipython().system('pip install -q google-cloud-storage==1.43.0')
get_ipython().system('pip install -q graphframes')


# In[3]:


import pyspark
import sys
from collections import Counter, OrderedDict, defaultdict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from time import time
from pathlib import Path
import pickle
import pandas as pd
from google.cloud import storage

import hashlib
def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

nltk.download('stopwords')


# In[4]:


# if nothing prints here you forgot to include the initialization script when starting the cluster
get_ipython().system('ls -l /usr/lib/spark/jars/graph*')


# In[5]:


from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf, SparkFiles
from pyspark.sql import SQLContext
from graphframes import *


# In[6]:


spark


# In[7]:


# Put your bucket name below and make sure you can access it without an error
import fnmatch
# Put your bucket name below and make sure you can access it without an error
bucket_name = 'ori318194230'
full_path = f"gs://{bucket_name}/"
paths=[]

client = storage.Client()
blobs = client.list_blobs(bucket_name)
for b in blobs:
    if b.name != 'graphframes.sh' and not fnmatch.fnmatch(b.name,'postings_gcp_anchor*')and not fnmatch.fnmatch(b.name,'pr*'):
        paths.append(full_path+b.name)


# ***GCP setup is complete!*** If you got here without any errors you've earned 10 out of the 35 points of this part.

# # Building an inverted index

# Here, we read the entire corpus to an rdd, directly from Google Storage Bucket and use your code from Colab to construct an inverted index.

# In[8]:


parquetFile = spark.read.parquet(*paths)
doc_text_pairs = parquetFile.select("text", "id").rdd


# We will count the number of pages to make sure we are looking at the entire corpus. The number of pages should be more than 6M

# In[9]:


# Count number of wiki pages
parquetFile.count()


# Let's import the inverted index module. Note that you need to use the staff-provided version called `inverted_index_gcp.py`, which contains helper functions to writing and reading the posting files similar to the Colab version, but with writing done to a Google Cloud Storage bucket.

# In[10]:


# if nothing prints here you forgot to upload the file inverted_index_gcp.py to the home dir
get_ipython().run_line_magic('cd', '-q /home/dataproc')
get_ipython().system('ls inverted_index_gcp.py')


# In[11]:


# adding our python module to the cluster
sc.addFile("/home/dataproc/inverted_index_gcp.py")
sys.path.insert(0,SparkFiles.getRootDirectory())


# In[12]:


from inverted_index_gcp import InvertedIndex


# **YOUR TASK (10 POINTS)**: Use your implementation of `word_count`, `reduce_word_counts`, `calculate_df`, and `partition_postings_and_write` functions from Colab to build an inverted index for all of English Wikipedia in under 2 hours.
# 
# A few notes: 
# 1. The number of corpus stopwords below is a bit bigger than the colab version since we are working on the whole corpus and not just on one file.
# 2. You need to slightly modify your implementation of  `partition_postings_and_write` because the signature of `InvertedIndex.write_a_posting_list` has changed and now includes an additional argument called `bucket_name` for the target bucket. See the module for more details.
# 3. You are not allowed to change any of the code not coming from Colab. 

# In[13]:




# PLACE YOUR CODE HERE
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links", 
                    "may", "first", "see", "history", "people", "one", "two", 
                    "part", "thumb", "including", "second", "following", 
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

NUM_BUCKETS = 124
def token2bucket_id(token):
  return int(_hash(token),16) % NUM_BUCKETS

# PLACE YOUR CODE HERE
def word_count(text, id):
  ''' Count the frequency of each word in `text` (tf) that is not included in 
  `all_stopwords` and return entries that will go into our posting lists. 
  Parameters:
  -----------
    text: str
      Text of one document
    id: int
      Document id
  Returns:
  --------
    List of tuples
      A list of (token, (doc_id, tf)) pairs 
      for example: [("Anarchism", (12, 5)), ...]
  '''
  tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
  # YOUR CODE HERE
  tokens = [token for token  in tokens if token not in all_stopwords]
  dict1={}
  list1=[]
  for tok in tokens :
    if tok in dict1:
      dict1[tok]=dict1[tok]+1
    else:
      dict1[tok]=1
  for k,v in dict1.items():
    list1.append((k,(id,v)))
  return list1

def reduce_word_counts(unsorted_pl):
  ''' Returns a sorted posting list by wiki_id.
  Parameters:
  -----------
    unsorted_pl: list of tuples
      A list of (wiki_id, tf) tuples 
  Returns:
  --------
    list of tuples
      A sorted posting list.
  '''
  # YOUR CODE HERE
  return sorted(unsorted_pl)
  
def calculate_df(postings):
  ''' Takes a posting list RDD and calculate the df for each token.
  Parameters:
  -----------
    postings: RDD
      An RDD where each element is a (token, posting_list) pair.
  Returns:
  --------
    RDD
      An RDD where each element is a (token, df) pair.
  '''
  # YOUR CODE HERE
  def f(postings): return(len(postings))
  return postings.mapValues(f)
  
def partition_postings_and_write(postings):
  ''' A function that partitions the posting lists into buckets, writes out 
  all posting lists in a bucket to disk, and returns the posting locations for 
  each bucket. Partitioning should be done through the use of `token2bucket` 
  above. Writing to disk should use the function  `write_a_posting_list`, a 
  static method implemented in inverted_index_colab.py under the InvertedIndex 
  class. 
  Parameters:
  -----------
    postings: RDD
      An RDD where each item is a (w, posting_list) pair.
  Returns:
  --------
    RDD
      An RDD where each item is a posting locations dictionary for a bucket. The
      posting locations maintain a list for each word of file locations and 
      offsets its posting list was written to. See `write_a_posting_list` for 
      more details.
  '''
  # YOUR CODE HERE


  return postings.map(lambda x:(token2bucket_id(x[0]),(x[0],x[1]))).groupByKey().map(lambda x:InvertedIndex.write_a_posting_list(x,bucket_name))


# In[ ]:


# time the index creation time
t_start = time()
# word counts map
word_counts = doc_text_pairs.flatMap(lambda x: word_count(x[0], x[1]))
postings = word_counts.groupByKey().mapValues(reduce_word_counts)
# filtering postings and calculate df
postings_filtered = postings.filter(lambda x: len(x[1])>50)
w2df = calculate_df(postings_filtered)
w2df_dict = w2df.collectAsMap()
# partition posting lists and write out
_ = partition_postings_and_write(postings_filtered).collect()
index_const_time = time() - t_start


# In[ ]:


# test index construction time
assert index_const_time < 60*120


# In[ ]:


# collect all posting lists locations into one super-set
super_posting_locs = defaultdict(list)
for blob in client.list_blobs(bucket_name, prefix='postings_gcp_text'):
  if not blob.name.endswith("pickle"):
    continue
  with blob.open("rb") as f:
    posting_locs = pickle.load(f)
    for k, v in posting_locs.items():
      super_posting_locs[k].extend(v)


# Putting it all together

# In[ ]:


from collections import Counter, defaultdict
import math
#create DL 
DL = {}
nf={}
doc_id_counts = postings_filtered.flatMap(lambda x: [(doc_id, tf) for doc_id, tf in x[1]]).reduceByKey(lambda x, y: x + y)
DL = doc_id_counts.collectAsMap()
nf_id= postings_filtered.flatMap(lambda x: [(doc_id, tf * tf) for doc_id, tf in x[1]]).reduceByKey(lambda x, y: x + y)
nf = nf_id.collectAsMap()

# Create inverted index instance
inverted = InvertedIndex()
# Adding the posting locations dictionary to the inverted index
inverted.posting_locs = super_posting_locs
# Add the token - df dictionary to the inverted index
inverted.df = w2df_dict
inverted.DL=DL
inverted.nf=nf
# write the global stats out
inverted.write_index('.', 'index_text')
# upload to gs
index_src = "index_text.pkl"
index_dst = f'gs://{bucket_name}/postings_gcp_text/{index_src}'
get_ipython().system('gsutil cp $index_src $index_dst')


# In[19]:


get_ipython().system('gsutil ls -lh $index_dst')




