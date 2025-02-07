from pathlib import Path

import gensim.downloader as api
import joblib
import numpy as np
import torch

from config import (
    LSTM_HIDDEN_DIM_DOC,
    LSTM_HIDDEN_DIM_QUERY,
    MODEL_PATH,
    MODEL_SIZE,
    OUTPUT_DIM,
    VOCAB_PATH,
    WORD2VEC_MODEL,
)
from model_architecture import TwoTowerModel
from preprocess import convert_to_indices, pad_truncate

# MODEL_PATH = "./model_twotower1.pth"
# VOCAB_PATH = "./words_to_ids.pkl"
# MODEL_SIZE = 300
# OUTPUT_DIM = 128
# LSTM_HIDDEN_DIM_QUERY = 128
# LSTM_HIDDEN_DIM_DOC = 256
# WORD2VEC_MODEL = "word2vec-google-news-300"


def make_embedding_tensor(word2idx: dict[str, int]) -> torch.Tensor:
    print("Loading Word2Vec model...")
    word2vec_model = api.load(WORD2VEC_MODEL)
    vocab_size = len(word2idx)
    embedding_matrix = np.zeros((vocab_size, MODEL_SIZE), dtype=np.float32)

    oov_count = 0
    for word, idx in word2idx.items():
        if word in word2vec_model:
            embedding_matrix[idx] = word2vec_model[word]  # type: ignore
        else:
            oov_count += 1

    print(
        f"Embedding matrix generated loaded. {(oov_count / vocab_size) * 100} % of words were out of vocab"
    )

    return torch.tensor(embedding_matrix)


def load_model() -> tuple[TwoTowerModel, dict[str, int]]:
    print(f"Loading TwoTowerModel")
    word2idx: dict[str, int] = joblib.load(Path(VOCAB_PATH))
    embeddings_tensor = make_embedding_tensor(word2idx)

    model = TwoTowerModel(
        output_dim=OUTPUT_DIM,
        embed_dim=MODEL_SIZE,
        lstm_hidden_dim_query=LSTM_HIDDEN_DIM_QUERY,
        lstm_hidden_dim_doc=LSTM_HIDDEN_DIM_DOC,
        embeddings_tensor=embeddings_tensor,
    )
    state_dict = torch.load(f=MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return model, word2idx
