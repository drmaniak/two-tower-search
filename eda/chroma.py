from langchain.vectorstores import Chroma
from utils import tokenize
from nltk.stem import PorterStemmer
import joblib
import constants
from typing import *
from langchain_chroma import Chroma
import gensim.downloader as api
from tqdm import tqdm
import numpy as np

class Embeddings:
    def __init__(self):
        self.punctuation_map = constants.punctuation_map
        self.model_size = 300

    def embed_documents(self, texts: List[str]) -> list:
        print('texts: ', texts)
        res = []
        for text in tqdm(texts):
            tokenized = tokenize(text, punctuation_map=punctuation_map, stemmer=PorterStemmer(), junk_punctuations=True)
            res.append(self.average_pooling(tokenized))
        return res
    
    def embed_query(self, query: str) -> list:
        tokenized = tokenize(query, punctuation_map=punctuation_map, stemmer=PorterStemmer(), junk_punctuations=True)
        return self.average_pooling(tokenized)

    def average_pooling(self, tokens: List[str]):
        embed = np.zeros((1, self.model_size))
        size = len(tokens)
        for t in tokens:
            if t in model:
                embed+=model[t]
        res = embed/max(size, 1)
        return res.tolist()[0]
    
# Specify the directory where your Chroma DB is stored.
persist_directory = "./chroma_langchain_db"  # adjust this path if necessary

# Optionally, specify a collection name if you have multiple collections.
collection_name = "example_collection"  # use your actual collection name if needed

#simple embeddings function
model = api.load("word2vec-google-news-300")
punctuation_map = constants.punctuation_map
model_size = 300

emb = Embeddings()

# Create (or load) the Chroma vector store.
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=emb,
    collection_name=collection_name  # this parameter is optional
)


query = "What did I have for breakfast this morning?"
results = vectorstore.similarity_search(query, k=2)
for doc in results:
    print(doc.page_content)