import argparse
import json
import shutil
from pathlib import Path

import joblib
import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

from config import BATCH_SIZE, CHROMA_DB_NAME, CHROMA_DB_PATH, DATA_PATH
from database_embedding import Embeddings
from model import load_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the langchain chroma db by specifying the percentage of training data to use"
    )

    parser.add_argument(
        "--truncate-data",
        type=float,
        default=1,
        help="Specify a percentage of the data to import (0-1)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load trained model & vocab
    model, word2idx = load_model()
    embedding_function = Embeddings(model, word2idx)

    if Path(CHROMA_DB_PATH).exists():
        print(f"Removing existing Chroma DB at : {CHROMA_DB_PATH}")
        shutil.rmtree(Path(CHROMA_DB_PATH))
        print("Creating new DB")
    else:
        print("No Chroma DB found, Creating one now.")

    # Load training data
    with open(DATA_PATH, "r") as file:
        training_data = json.load(file)
        truncate_len = int(round(len(training_data) * args.truncate_data))
        # Ensure truncate_len is a multiple of BATCH_SIZE
        truncate_len = (truncate_len // BATCH_SIZE) * BATCH_SIZE
        training_data = training_data[:truncate_len]

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

    # Process documents in batches
    for i, dd in tqdm(enumerate(training_data), total=len(training_data)):
        doc_text = dd["positive"]
        documents.append(
            Document(page_content=doc_text, metadata={"source": "search"}, id=str(i))
        )
        ids.append(str(i))

        # Every BATCH_SIZE docs, add to ChromaDB and reset batch lists
        if len(documents) >= BATCH_SIZE:
            vector_store.add_documents(documents=documents, ids=ids)
            documents.clear()  # Reset the batch
            ids.clear()

    # Add remaining documents if any
    if documents:
        vector_store.add_documents(documents=documents, ids=ids)

    print(f"Precomputed ChromaDB and stored at: {CHROMA_DB_PATH}")


if __name__ == "__main__":
    main()
