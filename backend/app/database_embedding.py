import numpy as np
import torch

from config import BATCH_SIZE, CONTEXT_LEN, MODEL_SIZE, QUERY_LENGTH
from model_architecture import TwoTowerModel
from preprocess import convert_to_indices, pad_truncate


class Embeddings:
    def __init__(self, model: TwoTowerModel, word2idx: dict[str, int]):
        self.model_size = MODEL_SIZE
        self.model = model
        self.word2idx = word2idx
        self.query_lengths = torch.tensor(
            [QUERY_LENGTH] * BATCH_SIZE, dtype=torch.int64
        )
        self.doc_lengths = torch.tensor([CONTEXT_LEN] * BATCH_SIZE, dtype=torch.int64)

    def embed_documents(self, texts: list[str]) -> list[torch.Tensor]:
        res = []
        # go in batches of batch_size
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            docs = [pad_truncate(doc, CONTEXT_LEN) for doc in batch]
            tokens = [convert_to_indices(doc, self.word2idx) for doc in docs]
            res.extend(self.embeddings(tokens))
        return res

    def embed_query(self, query: str) -> torch.Tensor:
        doc_padded = pad_truncate(query, CONTEXT_LEN)
        tokens = convert_to_indices(doc_padded, self.word2idx)
        tokens = [tokens] * BATCH_SIZE
        return self.embeddings(tokens)[0]

    def embeddings(self, tokens: list[list[int]]) -> np.ndarray:
        "We don't actually have a query or negative document or positive document"
        tokens = torch.tensor(tokens, dtype=torch.int64)  # type: ignore
        _, documents_embed, _ = self.model(
            tokens, tokens, tokens, self.query_lengths, self.doc_lengths
        )
        return documents_embed.detach().cpu().numpy()
