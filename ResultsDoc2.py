import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib
import numpy as np
from typing import *
from nltk.tokenize import word_tokenize
from langchain_chroma import Chroma
import gensim.downloader as api
from tqdm import tqdm
from langchain_core.documents import Document
from uuid import uuid4
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
from datasets import load_dataset

random.seed(42)

# Assuming the model has already been loaded and your TwoTowerModel class is available
# Replace this with your actual trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
word_to_ids = joblib.load("word_to_ids_v1.1.pkl")
#model_word2vec = api.load("word2vec-google-news-300")
embeds = joblib.load("embeds_v1.1.pkl")
print("Finished Loading")
# Load your trained model

class RNNTower(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNTower, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(device)
        #if x.dim() == 2:
            #x = x.unsqueeze(2)  # Add a dimension for input_size if missing
        #lengths = torch.tensor(lengths).long().to(device)
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hn = self.rnn(packed_x, h0)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        out = self.fc(out[torch.arange(out.shape[0]), lengths - 1])  # Get last valid output
        return out


class TwoTowerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(TwoTowerModel, self).__init__()
        self.query_tower = RNNTower(input_size, hidden_size, output_size, num_layers)
        self.doc_tower = RNNTower(input_size, hidden_size, output_size, num_layers)

    def forward(self, query, query_lengths, pos_doc, pos_doc_lengths, neg_doc, neg_doc_lengths):
        query_embed = self.query_tower(query, query_lengths)
        pos_doc_embed = self.doc_tower(pos_doc, pos_doc_lengths)
        neg_doc_embed = self.doc_tower(neg_doc, neg_doc_lengths)
        return query_embed, pos_doc_embed, neg_doc_embed

# Now for the custom embedding class
class Embeddings:
    def __init__(self, model: nn.Module):
        self.model = model
        self.model_size = 64  # The output size of the model (query and doc towers)
        self.device = device

    def embed_documents(self, texts: List[str]) -> list:
        """Embed documents using doc_tower"""
        #print('texts: ', texts)
        res = []
        for text in tqdm(texts):
            tokenized = word_tokenize(text)
            doc_embedding = self.embed_document(tokenized)
            res.append(doc_embedding)
        return res

    def embed_query(self, query: str) -> list:
        """Embed query using query_tower"""
        tokenized = word_tokenize(query)
        return self.embed_query_vector(tokenized)

    def embed_document(self, tokens: List[str]) -> torch.Tensor:
        """Convert document tokens into an embedding using doc_tower"""
        doc_emb, doc_len = self.text_to_ids(tokens, max_len=70)
        with torch.no_grad():
            doc_embedding = self.model.doc_tower(doc_emb.to(self.device), doc_len)
            doc_embedding = doc_embedding.squeeze()
        return doc_embedding.cpu().numpy().tolist()

    def embed_query_vector(self, tokens: List[str]) -> torch.Tensor:
        """Convert query tokens into an embedding using query_tower"""
        query_emb, query_len = self.text_to_ids(tokens, max_len=20)
        with torch.no_grad():
            query_embedding = self.model.query_tower(query_emb.to(self.device), query_len)
            query_embedding = query_embedding.squeeze()
        return query_embedding.cpu().numpy().tolist()

    def text_to_ids(self, tokens: List[str], max_len: int) -> torch.Tensor:
        """Convert tokens to token IDs and pad sequences"""
        token_ids = [word_to_ids.get(token, 0) for token in tokens]  # Default to 0 if not found
        if len(token_ids) == 0:
            token_ids = [0]  # Add a dummy token (e.g., padding token)

        embeddings = [torch.from_numpy(embeds[token_id]).float() for token_id in token_ids]  # Get embeddings for each token
        
        # Now pad the sequence to max_len (padding is done after embeddings are created)
        if len(embeddings) < max_len:
            padding = [torch.zeros(300)] * (max_len - len(embeddings))  # Pad with zero vectors
            embeddings.extend(padding)
        else:
            embeddings = embeddings[:max_len]  # Truncate to max_len if sequence is too long

        return torch.stack(embeddings, dim=0).unsqueeze(0), torch.tensor(min(len(token_ids),max_len)).unsqueeze(0)

model = TwoTowerModel(input_size=300, hidden_size=256, output_size=64).to(device)
model.load_state_dict(torch.load("twotower_model.pth"))
model.eval()
# Initialize the embeddings
emb = Embeddings(model = model)

# Initialize Chroma vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=emb,
    persist_directory="./chroma_langchain_db",  # Optional: persist data locally
    collection_metadata={"hnsw:space": "cosine"}
)

ds = load_dataset("microsoft/ms_marco", "v1.1")

train = ds['train']

# Define documents (for the sake of example)
documents = []
uuids = []
ids = 0
for i in tqdm(range(len(train)//100)):
    for k, passage in enumerate(ds['train'][i]['passages']['passage_text']):
        ids += 1
        document = Document(page_content=passage, metadata={"source": "search"}, id=ids)
        documents.append(document)
        uuids.append(str(ids))
        #sample['positive_url'] = ds['validation'][i]['passages']['url'][k]

# Add documents to the vector store
vector_store.add_documents(documents=documents, ids=uuids)
 
# Perform similarity search
query = "I have a fever and a high temperature. I am coughing. What do I have?"
print("Query:", query)
results = vector_store._collection.query(
    query_embeddings=[emb.embed_query(query)],  # Get query embedding
    n_results=5,  # Get top 5 most similar results
    include=["documents", "metadatas", "distances"]  # Include similarity scores
)

# Print results with similarity scores
for doc, metadata, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
    print(f"Document: {doc}\nMetadata: {metadata}\nSimilarity Score (Distance): {distance}\n")