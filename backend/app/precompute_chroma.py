import json

import joblib
import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

from config import BATCH_SIZE, CHROMA_DB_NAME, CHROMA_DB_PATH, DATA_PATH
from database_embedding import Embeddings
from model import load_model

# Load trained model & vocab
model, word2idx = load_model()
embedding_function = Embeddings(model, word2idx)

# Load training data
with open(DATA_PATH, "r") as file:
    training_data = json.load(file)
    training_data = training_data[:640]

print(f"Total # of Documents: {len(training_data)}")

# Initialize ChromaDB
vector_store = Chroma(
    collection_name=CHROMA_DB_NAME,
    embedding_function=embedding_function,  # type: ignore
    persist_directory=CHROMA_DB_PATH,
)


print(f"Preparing and adding documents in batches of {BATCH_SIZE}...")
documents = []
ids = []

# Truncate training_data to ensure it’s a multiple of BATCH_SIZE
training_data = training_data[: len(training_data) // BATCH_SIZE * BATCH_SIZE]
print(f"Total documents after truncation: {len(training_data)}")

batches = 0
# Process documents in batches
for i, dd in tqdm(enumerate(training_data), total=len(training_data)):
    doc_text = dd["positive"]
    documents.append(
        Document(page_content=doc_text, metadata={"source": "search"}, id=str(i))
    )
    ids.append(str(i))

    # Every BATCH_SIZE docs, add to ChromaDB and reset batch lists
    if len(documents) >= BATCH_SIZE:
        batches += 1
        vector_store.add_documents(documents=documents, ids=ids)
        documents.clear()  # Reset the batch
        ids.clear()

    if batches == 500:
        break

# Add remaining documents if any
if documents:
    vector_store.add_documents(documents=documents, ids=ids)

print(f"Precomputed ChromaDB and stored at: {CHROMA_DB_PATH}")
