# wikipedia-Search: A Wikipedia Search Engine

This project is a part of a Data Retrieval course at the university, where the goal was to create a search engine that can retrieve relevant information from a corpus of Wikipedia articles. The data retrieval process was done using an inverted index, which was created in GCP by indexing the body, title, and anchor text of the Wikipedia articles. The search functionality was then implemented in Python using the Flask framework, utilizing the inverted index to retrieve results. To improve the efficiency of the search engine, we pre-calculated certain values such as document lengths and created dictionaries to store them, so they don't need to be calculated at runtime. Additionally, we used the PageRank algorithm to rank the search results and applied techniques like stopword removal to improve retrieval.

# Requirements
* Python 3.6 or later
* Flask
* NLTK
* gzip
* csv
* numpy
* pandas

# Usage
* Run the command python3 search_frontend.py to start the application
* The application will be running on http://External_IP:8080/ by default
* To issue a query, navigate to a URL like http://External_IP:8080/search?query=hello+world

# Data
* The corpus of documents is a collection of Wikipedia articles
* The inverted index , PageRank , Pageview data is pre-computed and included in the project files
* Additional data, such as document lengths,term frequency  is calculated before the application starts

# The project includes several files, each containing specific functionality:


# inverted_index_gcp.py :
This file contains the InvertedIndex class, which is used to create and use the inverted index for the search engine. The inverted index is a data structure that allows for efficient retrieval of documents that contain a given query term.

# search_frontend.py :
This file runs the engine and contains the main logic for handling search queries. In this file, we use all our data and methods to perform evaluations to return the most relevant Wikipedia pages for a given query.

# GUI
The GUI for the search engine is a user-friendly interface that allows users to easily search for information on the internet. It consists of a simple text box where users can enter their search query and a search button to initiate the search. 
The search results displayed on the GUI are a list of the most relevant documents, with the title of each document prominently displayed. The titles are clearly presented in a consistent format, making it easy for users to quickly scan and identify the information they are looking for.
![WhatsApp Image 2023-01-16 at 21 17 51](https://user-images.githubusercontent.com/103646836/212752266-37e28a35-9d9b-4f23-88ee-ca39f3e553dd.jpg)


# search_bm25() :
This method is used to retrieve search results using the BM25 ranking algorithm. It takes in the user's query, processes it by removing stopwords and retrieves the results from the inverted index by running on each term in the query, check if the term is in the corpus, get the posting list of the term, then use the BM25 ranking algorithm to calculate the score for each document in the posting list. Then it return the top 100 results.

# search_body() :
This method is used to retrieve search results using the TF-IDF and Cosine Similarity of the body of articles only. It takes in the user's query, processes it by tokenizing and removing stopwords. Then it calculate the TF-IDF and Cosine Similarity of each term in the query with the terms in the documents, and returns the top 100 results.

# search_title() :
This method is used to retrieve search results that contain a query word in the title of articles. It takes in the user's query, processes it by tokenizing and removing stopwords, then it retrieves the results by running on each term in the query and check if the term is in the title of the corpus, then it counts the number of distinct query words that appear in the title. Then it returns all the results (not just top 100) ordered from best to worst based on the number of distinct query words that appear in the title.

# search_anchor() :
This method is used to retrieve search results that contain a query word in the anchor text of articles. It takes in the user's query, processes it by tokenizing and removing stopwords, then it retrieves the results by running on each term in the query and check if the term is in the anchor text of the corpus, then it counts the number of query words that appear in the anchor text linking to the page. Then it returns all the results (not just top 100) ordered from best to worst based on the number of query words that appear in the anchor text linking to the page.

# search() :
This function is the main search function of the project that combines the results of three different search methods (search_bm25, search_title and search_anchor) to return a final list of search results. The function takes a query from the user via the URL and performs tokenization and stopword removal on the query . 
The function first calls the search_bm25, search_title and search_anchor methods to retrieve the results from each of these search methods. Then it creates a Counter object named "ranked" that will be used to store the final scores of each document.
The function then loops through the results of each search method and assigns a score to each document based on the rank of the document in that method. For the text_data, it assigns score x/(i+1), where x is 90. For the title_data and anchor_data, it assigns a score of 8 and 4.5 respectively.
Then the function loops through the ranked dictionary to consider the page rank and page views of the document and assigns a score to each document based on the page rank and page views of the document.
Finally, the function sorts the ranked dictionary based on the scores and returns the top 100 results as a JSON object.

# get_pagerank() :
The function is used to get the PageRank values for a list of provided wiki article IDs. It takes a json payload of the list of article ids and returns a list of floats that correspond to the PageRank scores of the provided article IDs.
 
 # get_pageview() :
The function is used to get the number of page views that each of the provided wiki articles had in August 2021. It takes a json payload of the list of article ids and returns a list of integers that correspond to the page view numbers from August 2021 for the provided list of article IDs.

