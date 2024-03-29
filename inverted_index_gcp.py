# import pyspark
import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
from time import time
from pathlib import Path
import pickle
from google.cloud import storage
from collections import defaultdict
from contextlib import closing


import itertools
from pathlib import Path
from google.cloud import storage

# Size of each block/file to write
BLOCK_SIZE = 1024 * 1024 * 100  # 100 MB

class MultiFileWriter:
    """ Sequential binary writer to multiple files of up to BLOCK_SIZE each. """

    def __init__(self, base_dir, name, bucket_name):
        """ Initializes a MultiFileWriter object.

        Args:
            base_dir (str): The base directory to write the files to.
            name (str): The name of the files to write.
            bucket_name (str): The name of the Google Cloud Storage bucket to upload to.
        """
        self._base_dir = Path(base_dir)
        self._name = name
        
        # Generator that yields file objects with increasing numbers in the file name
        self._file_gen = (open(self._base_dir / f'{name}_{i:03}.bin', 'wb') 
                          for i in itertools.count())
        self._f = next(self._file_gen)
        
        # Connect to Google Cloud Storage bucket
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def write(self, b):
        """ Writes bytes to the files in BLOCK_SIZE chunks.

        Args:
            b (bytes): The bytes to write to the files.

        Returns:
            A list of tuples, each containing the name of the file and the position of the bytes in the file.
        """
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
            
            # If the current file is full, close and open a new one
            if remaining == 0:  
                self._f.close()
                self.upload_to_gcp()                
                self._f = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE
                
            # Write bytes to file
            self._f.write(b[:remaining])
            locs.append((self._f.name, pos))
            b = b[remaining:]
        return locs

    def close(self):
        """ Closes the current file. """
        self._f.close()

    def upload_to_gcp(self):
        """ Uploads the current file to Google Cloud Storage. """
        file_name = self._f.name
        blob = self.bucket.blob(f"postings_gcp_title/{file_name}")
        blob.upload_from_filename(file_name)


        



class MultiFileReader:
    """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """

    def __init__(self):
        """ Initializes a MultiFileReader object. """
        self._open_files = {}

    def read(self, locs, n_bytes, base_dir=""):
        """ Reads bytes from the files at specified locations.

        Args:
            locs (list): A list of tuples, each containing the name of the file and the position of the bytes in the file.
            n_bytes (int): The number of bytes to read.
            base_dir (str, optional): The base directory where the files are located. Defaults to "".

        Returns:
            The bytes read from the files.
        """
        b = []
        for f_name, offset in locs:
            # If file is not open yet, open it
            if f_name not in self._open_files:
                self._open_files[f_name] = open(base_dir + f_name, 'rb')

            # Seek to the position in the file to read from
            f = self._open_files[f_name]
            f.seek(offset)
            
            # Read up to BLOCK_SIZE bytes, or n_bytes if less than BLOCK_SIZE remaining
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)

    def close(self):
        """ Closes all open files. """
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        """ Called when exiting the 'with' block. Closes all open files. """
        self.close()
        return False



from collections import defaultdict
from contextlib import closing

TUPLE_SIZE = 6       # We're going to pack the doc_id and tf values in this 
                     # many bytes.
TF_MASK = 2 ** 16 - 1 # Masking the 16 low bits of an integer


class InvertedIndex:  
    def __init__(self, docs={}):
        """ Initializes the inverted index and add documents to it (if provided).
        Parameters:
        -----------
          docs: dict mapping doc_id to list of tokens
        """
        # stores document frequency per term
        self.df = Counter()
        # stores total frequency per term
        self.term_total = Counter()
        # stores posting list per term while building the index (internally), 
        # otherwise too big to store in memory.
        self._posting_list = defaultdict(list)

        self.DL={}# Dict key=doc_id , value=len(doc_id)

        self.nf={}

        self.id_title_dict={}

        # mapping a term to posting file locations, which is a list of 
        # (file_name, offset) pairs. Since posting lists are big we are going to
        # write them to disk and just save their location in this list. We are 
        # using the MultiFileWriter helper class to write fixed-size files and store
        # for each term/posting list its list of locations. The offset represents 
        # the number of bytes from the beginning of the file where the posting list
        # starts. 
        self.posting_locs = defaultdict(list)

        for doc_id, tokens in docs.items():
            self.add_doc(doc_id, tokens)

    def add_doc(self, doc_id, tokens):
        """ Adds a document to the index with a given `doc_id` and tokens. It counts
            the tf of tokens, then update the index (in memory, no storage 
            side-effects).
        """
        w2cnt = Counter(tokens)
        self.term_total.update(w2cnt)
        for w, cnt in w2cnt.items():
            self.df[w] = self.df.get(w, 0) + 1
            self._posting_list[w].append((doc_id, cnt))

    def write_index(self, base_dir, name):
        """ Write the in-memory index to disk. Results in the file: 
            (1) `name`.pkl containing the global term stats (e.g. df).
        """
        #### GLOBAL DICTIONARIES ####
        self._write_globals(base_dir, name)

    def _write_globals(self, base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'wb') as f:
            pickle.dump(self, f)

    def __getstate__(self):
        """ Modify how the object is pickled by removing the internal posting lists
            from the object's state dictionary. 
        """
        state = self.__dict__.copy()
        del state['_posting_list']
        return state

    def posting_lists_iter(self):
        """ A generator that reads one posting list from disk and yields 
            a (word:str, [(doc_id:int, tf:int), ...]) tuple.
        """
        with closing(MultiFileReader()) as reader:
            for w, locs in self.posting_locs.items():
                b = reader.read(locs[0], self.df[w] * TUPLE_SIZE)
                posting_list = []
                for i in range(self.df[w]):
                    doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
                    tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
                    posting_list.append((doc_id, tf))
                yield w, posting_list

    @staticmethod
    def read_index(base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def delete_index(base_dir, name):
        path_globals = Path(base_dir) / f'{name}.pkl'
        path_globals.unlink()
        for p in Path(base_dir).rglob(f'{name}_*.bin'):
            p.unlink()


    @staticmethod
def write_a_posting_list(b_w_pl, bucket_name):
    # A static method to write a posting list to a binary file and upload it to Google Cloud Storage.
    posting_locs = defaultdict(list)
    bucket_id, list_w_pl = b_w_pl
    
    # create a MultiFileWriter object to write the posting list to binary files
    with closing(MultiFileWriter(".", bucket_id, bucket_name)) as writer:
        for w, pl in list_w_pl: 
            # convert the doc_id and tf into bytes
            b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                          for doc_id, tf in pl])
            # write to file(s)
            locs = writer.write(b)
            # save file locations to index
            posting_locs[w].extend(locs)
        
        # upload the binary file(s) to Google Cloud Storage
        writer.upload_to_gcp() 
        # save the file locations to a pickle file and upload it to Google Cloud Storage
        InvertedIndex._upload_posting_locs(bucket_id, posting_locs, bucket_name)
    
    # return the bucket_id of the posting list
    return bucket_id


@staticmethod
def _upload_posting_locs(bucket_id, posting_locs, bucket_name):
    # A static method to save the file locations of a posting list to a pickle file and upload it to Google Cloud Storage.
    # save the file locations to a pickle file
    with open(f"{bucket_id}_posting_locs.pickle", "wb") as f:
        pickle.dump(posting_locs, f)
    
    # upload the pickle file to Google Cloud Storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob_posting_locs = bucket.blob(f"postings_gcp_title/{bucket_id}_posting_locs.pickle")
    blob_posting_locs.upload_from_filename(f"{bucket_id}_posting_locs.pickle")


