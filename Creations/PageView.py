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
paths = []

client = storage.Client()
blobs = client.list_blobs(bucket_name)
for b in blobs:
    if b.name != 'graphframes.sh' and not fnmatch.fnmatch(b.name, 'postings_gcp_text*') and not fnmatch.fnmatch(b.name, 'postings_gcp_anchor*') and not fnmatch.fnmatch(
            b.name, 'pr*'):
        paths.append(full_path + b.name)

# Page View

pv_path = 'https://dumps.wikimedia.org/other/pageview_complete/monthly/2021/2021-08/pageviews-202108-user.bz2'
p = Path(pv_path)
pv_name = p.name
pv_temp = f'{p.stem}-4dedup.txt'
pv_clean = f'{p.stem}.pkl'
# Download the file (2.3GB)
!wget -N $pv_path
# Filter for English pages, and keep just two fields: article ID (3) and monthly
# total number of page views (5). Then, remove lines with article id or page
# view values that are not a sequence of digits.
!bzcat $pv_name | grep "^en\.wikipedia" | cut -d' ' -f3,5 | grep -P "^\d+\s\d+$" > $pv_temp
# Create a Counter (dictionary) that sums up the pages views for the same
# article, resulting in a mapping from article id to total page views.
wid2pv = Counter()
with open(pv_temp, 'rt') as f:
  for line in f:
    parts = line.split(' ')
    wid2pv.update({int(parts[0]): int(parts[1])})
# write out the counter as binary file (pickle it)
with open(pv_clean, 'wb') as f:
  pickle.dump(wid2pv, f)
# read in the counter
with open(pv_clean, 'rb') as f:
  wid2pv = pickle.loads(f.read())