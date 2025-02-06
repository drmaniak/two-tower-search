import joblib
import json
from typing import *
from langchain_chroma import Chroma
from tqdm import tqdm
from eda.preprocess import pad_truncate, convert_to_indices
import torch
from eda.two_tower_architecture import TwoTowerModel
from langchain_core.documents import Document
import numpy as np
import gensim.downloader as api

DATA_PATH = '/Users/yuliagoryachev/Documents/mlx/mlx_week2/two-tower-search/eda/train_triples_v1.1.json'
MODEL_PATH = '/Users/yuliagoryachev/Documents/mlx/mlx_week2/two-tower-search/eda/model_twotower1.pth'
VOCAB_PATH = '/Users/yuliagoryachev/Documents/mlx/mlx_week2/two-tower-search/eda/word_to_ids.pkl'

CONTEXT_LEN = 40
QUERY_LENGTH = 12
MODEL_SIZE = 300
OUTPUT_DIM = 128
LSTM_HIDDEN_DIM_QUERY = 128
LSTM_HIDDEN_DIM_DOC = 256
BATCH_SIZE = 64
VOCAB_SIZE = 607348

print('Load the data')
with open(DATA_PATH) as f:
    train = json.load(f)
    train = train[:640]

print('Load the word2idx')
word2idx = joblib.load(VOCAB_PATH)

#make lengths tensors
query_lengths = torch.tensor([QUERY_LENGTH]*BATCH_SIZE, dtype=torch.int64)
doc_lengths = torch.tensor([CONTEXT_LEN]*BATCH_SIZE, dtype=torch.int64)

def make_embedding_tensor(word2idx: dict) -> torch.Tensor:
    print('Load the word2vec model')
    word2vec_model = api.load("word2vec-google-news-300")
    vocab_size = len(word2idx)
    embedding_matrix = np.zeros((vocab_size, MODEL_SIZE), dtype=np.float32)

    for word, idx in word2idx.items():
        if word in word2vec_model:
            embedding_matrix[idx] = word2vec_model[word]
        else:
        
            if word == "<pad>":
                embedding_matrix[idx] = np.zeros(MODEL_SIZE, dtype=np.float32)
            else:
                embedding_matrix[idx] = np.zeros(MODEL_SIZE, dtype=np.float32)
    # Convert the numpy matrix to a torch tensor.
    embeddings_tensor = torch.tensor(embedding_matrix)
    return embeddings_tensor

print('Load the two tower model')
embeddings_tensor = make_embedding_tensor(word2idx)
model = TwoTowerModel(OUTPUT_DIM, MODEL_SIZE, LSTM_HIDDEN_DIM_QUERY, LSTM_HIDDEN_DIM_DOC, embeddings_tensor)
state_dict = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()


class Embeddings:
    def __init__(self):
        self.model_size = MODEL_SIZE

    def embed_documents(self, texts: List[str]) -> List[torch.Tensor]:
        # print('texts: ', texts)
        res = []
        # go in batches of batch_size
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            docs = [pad_truncate(doc, CONTEXT_LEN) for doc in batch]    
            tokens = [convert_to_indices(doc, word2idx) for doc in docs]
            res.extend(self.embeddings(tokens))
        return res
    
    def embed_query(self, query: str) -> torch.Tensor:
        doc_padded = pad_truncate(query, CONTEXT_LEN)
        tokens = convert_to_indices(doc_padded, word2idx)
        tokens = [tokens]*BATCH_SIZE
        return self.embeddings(tokens)[0]

    def embeddings(self, tokens: List[str]) -> np.ndarray:
        "We don't actually have a query or negative document or positive document"
        tokens = torch.tensor(tokens, dtype=torch.int64)
        _, documents_embed, _ = model(tokens, tokens, tokens, query_lengths, doc_lengths) 
        return documents_embed.detach().cpu().numpy()
    

if __name__ == "__main__":
    emb = Embeddings()
    vector_store = Chroma(
    collection_name="mlx_wk2_collection",
    embedding_function=emb,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )
    print('preparing documents')
    documents = []
    train = train[:len(train)//BATCH_SIZE*BATCH_SIZE]
    print('Len train: ', len(train))
    for i, dd in tqdm(enumerate(train)):
        doc = dd['positive']
        documents.append(Document(page_content=doc, metadata={"source": "search"}, id=i))

    print('adding documents')
    ids = [str(i) for i in range(len(documents))]

    vector_store.add_documents(documents=documents, ids=ids)
    question = "Are the inner workings of a rebuildable atomizer simple?"
    results = vector_store.similarity_search(
        question,
        k=2,
        filter={"source": "search"},
    )
    print('Question: ', question)
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")
        print()
    


