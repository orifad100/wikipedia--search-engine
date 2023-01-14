# wikipedia-Search: A Wikipedia Search Engine

This project is a search engine for a Wikipedia corpus built using the Google Cloud Platform (GCP) and Python. The search engine utilizes an inverted index, pre-processed data, and various data sources to return the most relevant Wikipedia pages for a given query.

#Requirements
* Python 3.6 or later
* Flask
* NLTK
* gzip
* csv
* numpy
* pandas

The project includes several files, each containing specific functionality:

1)inverted_index_gcp.py: This file contains the InvertedIndex class, which is used to create and use the inverted index for the search engine. The inverted index is a data structure that allows for efficient retrieval of documents that contain a given query term.

2)search_frontend.py: This file runs the engine and contains the main logic for handling search queries. In this file, we use all our data and methods to perform evaluations to return the most relevant Wikipedia pages for a given query.



